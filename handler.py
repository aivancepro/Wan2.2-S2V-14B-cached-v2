"""
RunPod Serverless Handler for Wan2.2-S2V-14B (Sound-to-Video)

Accepts audio + image input and generates synchronized video.

Uses RunPod model caching for faster cold starts.
Set Model field in endpoint config to: Wan-AI/Wan2.2-S2V-14B
"""

import os
import sys
import base64
import tempfile
import subprocess
from pathlib import Path

import runpod


# RunPod model cache configuration
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
MODEL_NAME = "Wan-AI/Wan2.2-S2V-14B"


def find_cached_model(model_name: str) -> str | None:
    """Find model in RunPod cache, return path or None."""
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[0])
    return None


# Model configuration - try cache first, fall back to bundled/env
MODEL_DIR = (
    find_cached_model(MODEL_NAME)
    or os.environ.get("MODEL_DIR", "/models/Wan2.2-S2V-14B")
)
print(f"Using model directory: {MODEL_DIR}")

DEFAULT_SIZE = os.environ.get("DEFAULT_SIZE", "832*480")  # width*height
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "30"))
OFFLOAD_MODEL = os.environ.get("OFFLOAD_MODEL", "True").lower() == "true"


def save_base64_to_file(b64_data: str, output_path: str) -> str:
    """Decode base64 data and save to file."""
    # Handle data URI format
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]

    decoded = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(decoded)
    return output_path


def encode_file_to_base64(file_path: str) -> str:
    """Read file and encode to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Wan2.2-S2V-14B.

    Input:
        image_base64: Base64 encoded input image (required)
        audio_base64: Base64 encoded audio file (required)
        prompt: Text prompt describing the scene (optional)
        negative_prompt: What to avoid (optional)
        size: Output resolution as "width*height" (default: 832*480)
        steps: Number of inference steps (default: 30)
        cfg: CFG scale (default: 5.0)
        seed: Random seed (default: -1 for random)
        pose_video_base64: Optional pose reference video (base64)

    Output:
        video: Base64 encoded MP4 video
    """
    job_input = job.get("input")

    # Health check - returns immediately without loading model
    # Used by RunPod's testing phase during deployment
    if job_input == "health_check" or (isinstance(job_input, dict) and job_input.get("health_check")):
        return {
            "status": "healthy",
            "model_dir": MODEL_DIR,
            "model_available": os.path.exists(MODEL_DIR),
            "message": "Handler ready. Model will be loaded on first inference request."
        }

    if not isinstance(job_input, dict):
        return {"error": "Invalid request: missing 'input' field"}

    # Validate required inputs
    if "image_base64" not in job_input:
        return {"error": "Missing required field: image_base64"}
    if "audio_base64" not in job_input:
        return {"error": "Missing required field: audio_base64"}

    # Lazy model download on first inference (image kept small; a module-load download of
    # ~40GB blew RunPod's worker-ready window -> crash loop). Slow only on the first cold job.
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        from huggingface_hub import snapshot_download
        print(f"Model absent at {MODEL_DIR} — downloading {MODEL_NAME} (~40GB)…", flush=True)
        snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)
        print("Model download complete.", flush=True)

    # Extract parameters
    image_b64 = job_input["image_base64"]
    audio_b64 = job_input["audio_base64"]
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted")
    size = job_input.get("size", DEFAULT_SIZE)
    steps = job_input.get("steps", DEFAULT_STEPS)
    cfg = job_input.get("cfg", 5.0)
    seed = job_input.get("seed", -1)
    pose_video_b64 = job_input.get("pose_video_base64")

    # Create temp directory for this job
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Detect image format from base64 header or default to jpg
        image_ext = ".jpg"
        if image_b64.startswith("data:image/png"):
            image_ext = ".png"
        elif image_b64.startswith("data:image/webp"):
            image_ext = ".webp"

        # Detect audio format
        audio_ext = ".wav"
        if "audio/mp3" in audio_b64 or "audio/mpeg" in audio_b64:
            audio_ext = ".mp3"

        # Save input files
        image_path = temp_path / f"input{image_ext}"
        audio_path = temp_path / f"input{audio_ext}"
        output_path = temp_path / "output.mp4"

        try:
            save_base64_to_file(image_b64, str(image_path))
            save_base64_to_file(audio_b64, str(audio_path))
        except Exception as e:
            return {"error": f"Failed to decode input: {e}"}

        # Handle optional pose video
        pose_path = None
        if pose_video_b64:
            pose_path = temp_path / "pose.mp4"
            try:
                save_base64_to_file(pose_video_b64, str(pose_path))
            except Exception as e:
                return {"error": f"Failed to decode pose video: {e}"}

        # Build command
        cmd = [
            sys.executable, "generate.py",
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", MODEL_DIR,
            "--image", str(image_path),
            "--audio", str(audio_path),
            "--save_file", str(output_path),
            "--sample_steps", str(steps),
            "--sample_guide_scale", str(cfg),
        ]

        if prompt:
            cmd.extend(["--prompt", prompt])

        # NB: Wan2.2 generate.py n'accepte PAS --negative_prompt (negative prompt
        # par défaut codé en dur dans le modèle) — ne pas le passer.

        if seed >= 0:
            cmd.extend(["--base_seed", str(seed)])

        if OFFLOAD_MODEL:
            cmd.extend(["--offload_model", "True"])

        if pose_path:
            cmd.extend(["--pose_video", str(pose_path)])

        # Run inference
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd="/app"  # Assuming model code is in /app
            )

            if result.returncode != 0:
                return {
                    "error": f"Generation failed: {result.stderr}",
                    "stdout": result.stdout
                }
        except subprocess.TimeoutExpired:
            return {"error": "Generation timed out (10 minutes)"}
        except Exception as e:
            return {"error": f"Generation failed: {e}"}

        # Find output video (generate.py may use different naming)
        output_files = list(temp_path.glob("*.mp4"))
        if not output_files:
            return {
                "error": "No output video generated",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        output_video = output_path if output_path.exists() else output_files[0]

        # Encode output video
        try:
            video_b64 = encode_file_to_base64(str(output_video))
        except Exception as e:
            return {"error": f"Failed to encode output: {e}"}

        return {"video": video_b64}


# For local testing
if __name__ == "__main__":
    # Test with sample input
    test_job = {
        "input": {
            "image_base64": "...",  # Add test image
            "audio_base64": "...",  # Add test audio
            "prompt": "a person talking",
        }
    }
    print(handler(test_job))
else:
    runpod.serverless.start({"handler": handler})
