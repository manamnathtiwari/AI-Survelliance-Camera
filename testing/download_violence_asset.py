import urllib.request
import os
import sys

def download_file(url, filename):
    try:
        print(f"Attempting to download {url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(filename, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Success: Downloaded to {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

if __name__ == "__main__":
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # 3. Violence Video Candidates
    # Trying known public repos for violence detection samples
    violence_urls = [
        "https://github.com/harshilpatel312/Violence-Detection-Technique/raw/master/violence_videos/V_11.mp4",
        "https://github.com/mohamedmustafa/Violence-Detection-Dataset/raw/main/videos/V_1.mp4", 
        "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/videos/fight/fi001.mp4"
    ]

    target_path = os.path.join(assets_dir, "sample_violence.mp4")
    
    success = False
    for url in violence_urls:
         if download_file(url, target_path):
             success = True
             break
    
    if not success:
        print("Error: Could not download any violence samples. Please manually add 'sample_violence.mp4' to testing/assets/")
        sys.exit(1)
