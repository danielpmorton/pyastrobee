"""Concatenating a sequence of recorded videos together"""

from typing import Union

import os
import re
from pathlib import Path

# Files created during the concatenation process
SORTING_FILE = "join_video.txt"
OUTPUT_VIDEO_FILE = "concatenated.mp4"


def concatenate_videos(directory: Union[str, Path], cleanup: bool = False) -> None:
    """Concatenates a sequence of videos into a single video

    - FFMPEG must be installed first
    - All videos must have the same streams (same codecs, same time base, etc.)
    - All videos in the directory should be named in alphanumeric order to match the order that they will be
      concatenated: e.g. "0.mp4", "1.mp4", "2.mp4", ...
    - The videos to concatenate should be the only items in the directory

    Args:
        directory (Union[str, Path]): Directory containing the videos to concatenate
        cleanup (bool, optional): Whether to delete the individual videos after concatenation. Defaults to False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory: {str(directory)} not recognized")
    video_order = _get_videos(directory)
    _ffmpeg_concat(directory, video_order)
    print(f"Concatenated video saved to {str(Path(directory, OUTPUT_VIDEO_FILE))}")
    if cleanup:
        _cleanup(directory, video_order)


## Helper functions below ##


# https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
def _sorted_alphanumeric(data: list[str]) -> list[str]:
    """Sort a list of strings via alphanumeric order

    Args:
        data (list[str]): Strings to sort

    Returns:
        list[str]: Alphanumerically-sorted data
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def _get_videos(directory: Path) -> list[str]:
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
    return _sorted_alphanumeric(os.listdir(str(directory)))


def _ffmpeg_concat(directory: Path, order: list[str]):
    """Run FFMPEG concatenation process

    Args:
        directory (Path): Directory of the videos to concatenate
        order (list[str]): Video names in the order of concatenation
    """
    # FFMPEG needs to create a txt file to store the order of the videos first
    vid_order_file = str(Path(directory, SORTING_FILE))
    with open(vid_order_file, "w") as f:
        for vid in order:
            f.write("file " + str(Path(directory, vid).resolve()) + "\n")
    output_file = str(Path(directory, OUTPUT_VIDEO_FILE))
    # Run the FFMPEG command
    os.system(f"ffmpeg -f concat -safe 0 -i {vid_order_file} -c copy {output_file}")


def _cleanup(directory: Path, videos: list[str]):
    """Delete all of the individual pre-concatenation videos, and the text file generated for video sorting

    Args:
        directory (Path): Directory containing the concatenated video files
        videos (list[str]): Video names to delete
    """
    for vid in videos:
        p = Path(directory, vid)
        p.unlink()
    Path(directory, SORTING_FILE).unlink()


def _main():
    """Run as a script from user input"""
    while True:
        dir_str = input("Provide a directory of videos to concatenate: \n")
        dir_path = Path(dir_str)
        if dir_path.exists():
            break
        print("Directory not recognized, try again")
    video_order = _get_videos(dir_path)
    _ffmpeg_concat(dir_path, video_order)
    input("Press Enter to clean up")
    _cleanup(dir_path, video_order)


if __name__ == "__main__":
    _main()
