"""Script for concatenating a sequence of recorded videos together

This requires that ffmpeg is installed

All videos should be named in alphanumeric order to match the order that they will be concatenated: e.g.
"0.mp4", "1.mp4", "2.mp4", ...

The videos to concatenate should be the only items in the directory
"""

import os
import re
from pathlib import Path

# Files created during the concatenation process
SORTING_FILE = "join_video.txt"
OUTPUT_VIDEO_FILE = "concatenated.mp4"


# https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
def sorted_alphanumeric(data: list[str]) -> list[str]:
    """Sort a list of strings via alphanumeric order

    Args:
        data (list[str]): Strings to sort

    Returns:
        list[str]: Alphanumerically-sorted data
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def get_directory() -> Path:
    """Get the directory of the videos to concatenate via user input

    Returns:
        Path: Path object to directory
    """
    while True:
        directory = input("Provide a directory of videos to concatenate: \n")
        dir_path = Path(directory)
        if dir_path.exists():
            break
        print("Directory not recognized, try again")
    return dir_path


def get_videos(directory: Path) -> list[str]:
    """Get a sorted list of video files in concatenation order, within the given directory

    Args:
        directory (Path): Directory of the videos to concatenate

    Raises:
        RuntimeError: If the files are named improperly (must be alphanumeric for sorting, and MP4 format)

    Returns:
        list[str]: Video names in the order of concatenation
    """
    for filename in os.listdir(str(directory)):
        p = Path(directory, filename)
        if p.name == SORTING_FILE:
            p.unlink()
            continue
        if p.name == OUTPUT_VIDEO_FILE:
            input(
                "WARNING: Video appears to be already concatenated. Press Enter to overwrite"
            )
            p.unlink()
            continue
        if p.suffix != ".mp4":
            raise RuntimeError("Invalid file extension: ", p.suffix, ". Expected .mp4")
        if not p.stem.isnumeric():
            raise RuntimeError(
                "Unrecognized file naming. Expect files named with their alphanumeric order"
            )
    return sorted_alphanumeric(os.listdir(str(directory)))


def concatenate(directory: Path, order: list[str]):
    vid_order_file = str(Path(directory, SORTING_FILE))
    with open(vid_order_file, "w") as f:
        for vid in order:
            f.write("file " + str(Path(directory, vid).resolve()) + "\n")

    output_file = str(Path(directory, OUTPUT_VIDEO_FILE))
    os.system(f"ffmpeg -f concat -safe 0 -i {vid_order_file} -c copy {output_file}")


def cleanup(directory: Path, videos: list[str]):
    for vid in videos:
        p = Path(directory, vid)
        p.unlink()
    Path(directory, SORTING_FILE).unlink()


def _main():
    dir_path = get_directory()
    video_order = get_videos(dir_path)
    concatenate(dir_path, video_order)
    input("Press Enter to clean up")
    cleanup(dir_path, video_order)


if __name__ == "__main__":
    _main()
