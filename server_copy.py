from functools import wraps
from flask import (
    Flask,
    jsonify,
    request,
    Response,
    render_template_string,
    abort,
    send_from_directory,
    send_file,
)
from flask_cors import CORS
from flask_compress import Compress
import markdown
import argparse
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BlipForConditionalGeneration
import unicodedata
import torch
import time
import os
import gc
import secrets
from PIL import Image
import base64
from io import BytesIO
from random import randint
import webuiapi
import hashlib
from constants import *
from colorama import Fore, Style, init as colorama_init

colorama_init()


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(
            namespace, self.dest, values.replace('"', "").replace("'", "").split(",")
        )


# Script arguments
parser = argparse.ArgumentParser(
    prog="SillyTavern Extras", description="Web API for transformers models"
)
parser.add_argument(
    "--port", type=int, help="Specify the port on which the application is hosted"
)
parser.add_argument(
    "--listen", action="store_true", help="Host the app on the local network"
)
parser.add_argument(
    "--share", action="store_true", help="Share the app on CloudFlare tunnel"
)
parser.add_argument("--cpu", action="store_true", help="Run the models on the CPU")
parser.add_argument("--cuda", action="store_false", dest="cpu", help="Run the models on the GPU")
parser.set_defaults(cpu=True)

parser.add_argument("--chroma-host", help="Host IP for a remote ChromaDB instance")
parser.add_argument("--chroma-port", help="HTTP port for a remote ChromaDB instance (defaults to 8000)")
parser.add_argument("--chroma-folder", help="Path for chromadb persistence folder", default='.chroma_db')
parser.add_argument('--chroma-persist', help="Chromadb persistence", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--secure", action="store_true", help="Enforces the use of an API key"
)



parser.add_argument(
    "--enable-modules",
    action=SplitArgs,
    default=[],
    help="Override a list of enabled modules",
)

args = parser.parse_args()

port = 7860
host = "0.0.0.0"


# sd_remote_ssl = args.sd_remote_ssl
# sd_remote_auth = args.sd_remote_auth

modules = (
    args.enable_modules if args.enable_modules and len(args.enable_modules) > 0 else []
)

if len(modules) == 0:
    print(
        f"{Fore.RED}{Style.BRIGHT}You did not select any modules to run! Choose them by adding an --enable-modules option"
    )
    print(f"Example: --enable-modules=caption,summarize{Style.RESET_ALL}")

# Models init
device_string = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
device = torch.device(device_string)
torch_dtype = torch.float32 if device_string == "cpu" else torch.float16

if not torch.cuda.is_available() and not args.cpu:
    print(f"{Fore.YELLOW}{Style.BRIGHT}torch-cuda is not supported on this device. Defaulting to CPU mode.{Style.RESET_ALL}")

print(f"{Fore.GREEN}{Style.BRIGHT}Using torch device: {device_string}{Style.RESET_ALL}")



if "tts" in modules:
    print("tts module is deprecated. Please use silero-tts instead.")
    modules.remove("tts")
    modules.append("silero-tts")


if "silero-tts" in modules:
    if not os.path.exists(SILERO_SAMPLES_PATH):
        os.makedirs(SILERO_SAMPLES_PATH)
    print("Initializing Silero TTS server")
    from silero_api_server import tts

    tts_service = tts.SileroTtsService(SILERO_SAMPLES_PATH)
    if len(os.listdir(SILERO_SAMPLES_PATH)) == 0:
        print("Generating Silero TTS samples...")
        tts_service.update_sample_text(SILERO_SAMPLE_TEXT)
        tts_service.generate_samples()


if "edge-tts" in modules:
    print("Initializing Edge TTS client")
    import tts_edge as edge


if "chromadb" in modules:
    print("Initializing ChromaDB")
    import chromadb
    import posthog
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    # Assume that the user wants in-memory unless a host is specified
    # Also disable chromadb telemetry
    posthog.capture = lambda *args, **kwargs: None
    if args.chroma_host is None:
        if args.chroma_persist:
            chromadb_client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory=args.chroma_folder, chroma_db_impl='duckdb+parquet'))
            print(f"ChromaDB is running in-memory with persistence. Persistence is stored in {args.chroma_folder}. Can be cleared by deleting the folder or purging db.")
        else:
            chromadb_client = chromadb.Client(Settings(anonymized_telemetry=False))
            print(f"ChromaDB is running in-memory without persistence.")
    else:
        chroma_port=(
            args.chroma_port if args.chroma_port else DEFAULT_CHROMA_PORT
        )
        chromadb_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                chroma_api_impl="rest",
                chroma_server_host=args.chroma_host,
                chroma_server_http_port=chroma_port
            )
        )
        print(f"ChromaDB is remotely configured at {args.chroma_host}:{chroma_port}")

    chromadb_embedder = SentenceTransformer(embedding_model)
    chromadb_embed_fn = lambda *args, **kwargs: chromadb_embedder.encode(*args, **kwargs).tolist()

    # Check if the db is connected and running, otherwise tell the user
    try:
        chromadb_client.heartbeat()
        print("Successfully pinged ChromaDB! Your client is successfully connected.")
    except:
        print("Could not ping ChromaDB! If you are running remotely, please check your host and port!")

# Flask init
app = Flask(__name__)
CORS(app)  # allow cross-domain requests
Compress(app) # compress responses
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


def require_module(name):
    def wrapper(fn):
        @wraps(fn)
        def decorated_view(*args, **kwargs):
            if name not in modules:
                abort(403, "Module is disabled by config")
            return fn(*args, **kwargs)

        return decorated_view

    return wrapper




    
ignore_auth = []    

api_key = os.environ.get("password")

def is_authorize_ignored(request):
    view_func = app.view_functions.get(request.endpoint)

    if view_func is not None:
        if view_func in ignore_auth:
            return True
    return False

# @app.before_request
# def before_request():
#     # Request time measuring
#     request.start_time = time.time()

#     # Checks if an API key is present and valid, otherwise return unauthorized
#     # The options check is required so CORS doesn't get angry
#     try:
#         if request.method != 'OPTIONS' and is_authorize_ignored(request) == False and getattr(request.authorization, 'token', '') != api_key:
#             print(f"WARNING: Unauthorized API key access from {request.remote_addr}")
#             response = jsonify({ 'error': '401: Invalid API key' })
#             response.status_code = 401
#             return "so"
#     except Exception as e:
#         print(f"API key check error: {e}")
#         return "so"

@app.before_request
def before_request():
    # Request time measuring
    request.start_time = time.time()

    return None


@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    response.headers["X-Request-Duration"] = str(duration)
    return response


@app.route("/", methods=["GET"])
def index():
    with open("./README.md", "r", encoding="utf8") as f:
        content = f.read()
    return render_template_string(markdown.markdown(content, extensions=["tables"]))


@app.route("/api/extensions", methods=["GET"])
def get_extensions():
    extensions = dict(
        {
            "extensions": [
                {
                    "name": "not-supported",
                    "metadata": {
                        "display_name": """<span style="white-space:break-spaces;">Extensions serving using Extensions API is no longer supported. Please update the mod from: <a href="https://github.com/Cohee1207/SillyTavern">https://github.com/Cohee1207/SillyTavern</a></span>""",
                        "requires": [],
                        "assets": [],
                    },
                }
            ]
        }
    )

    return jsonify(extensions)




@app.route("/api/modules", methods=["GET"])
def get_modules():
    return jsonify({"modules": modules})


@app.route("/api/tts/speakers", methods=["GET"])
@require_module("silero-tts")
def tts_speakers():
    voices = [
        {
            "name": speaker,
            "voice_id": speaker,
            "preview_url": f"{str(request.url_root)}api/tts/sample/{speaker}",
        }
        for speaker in tts_service.get_speakers()
    ]
    return jsonify(voices)


@app.route("/api/tts/generate", methods=["POST"])
@require_module("silero-tts")
def tts_generate():
    voice = request.get_json()
    if "text" not in voice or not isinstance(voice["text"], str):
        abort(400, '"text" is required')
    if "speaker" not in voice or not isinstance(voice["speaker"], str):
        abort(400, '"speaker" is required')
    # Remove asterisks
    voice["text"] = voice["text"].replace("*", "")
    try:
        audio = tts_service.generate(voice["speaker"], voice["text"])
        return send_file(audio, mimetype="audio/x-wav")
    except Exception as e:
        print(e)
        abort(500, voice["speaker"])


@app.route("/api/tts/sample/<speaker>", methods=["GET"])
@require_module("silero-tts")
def tts_play_sample(speaker: str):
    return send_from_directory(SILERO_SAMPLES_PATH, f"{speaker}.wav")


@app.route("/api/edge-tts/list", methods=["GET"])
@require_module("edge-tts")
def edge_tts_list():
    voices = edge.get_voices()
    return jsonify(voices)


@app.route("/api/edge-tts/generate", methods=["POST"])
@require_module("edge-tts")
def edge_tts_generate():
    data = request.get_json()
    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')
    if "voice" not in data or not isinstance(data["voice"], str):
        abort(400, '"voice" is required')
    if "rate" in data and isinstance(data['rate'], int):
        rate = data['rate']
    else:
        rate = 0
    # Remove asterisks
    data["text"] = data["text"].replace("*", "")
    try:
        audio = edge.generate_audio(text=data["text"], voice=data["voice"], rate=rate)
        return Response(audio, mimetype="audio/mpeg")
    except Exception as e:
        print(e)
        abort(500, data["voice"])


@app.route('/test')
def my_page():
    return 'this is test page..'

ignore_auth.append(tts_play_sample)
app.run(host=host, port=port)
