import urllib.request
import os

def download_file(url, filename):
    try:
        print(f"Downloading {url} to {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # 1. People Video (for Tracker, Object Det, Pose)
    v1_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    download_file(v1_url, os.path.join(assets_dir, "sample_people.mp4"))
    
    # 2. Face Image (Use a standard sample if possible, or extract from video later)
    # We will extract it in the test script.

    print("Assets setup check complete.")
