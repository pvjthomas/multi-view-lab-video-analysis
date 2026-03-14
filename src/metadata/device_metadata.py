import json
from pathlib import Path

class VideoMetadata:
    def __init__(self, metadata_file="data/metadata/videos.json"):
        self.metadata_file = Path(metadata_file)
        self.data = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.data = json.load(f)

    def add_video(self, filename, device_type, viewpoint, fps, resolution, timestamp):
        self.data[filename] = {
            "device_type": device_type,
            "viewpoint": viewpoint,
            "fps": fps,
            "resolution": resolution,
            "timestamp": timestamp
        }
        self._save()

    def get_metadata(self, filename):
        return self.data.get(filename, None)

    def _save(self):
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(self.data, f, indent=4)

if __name__ == "__main__":
    vm = VideoMetadata()
    vm.add_video(
        "lab_task1_gopro.mp4",
        device_type="GoPro Hero10",
        viewpoint="FPV",
        fps=60,
        resolution="1920x1080",
        timestamp="2026-03-14 10:23:00"
    )
    print(vm.get_metadata("lab_task1_gopro.mp4"))
