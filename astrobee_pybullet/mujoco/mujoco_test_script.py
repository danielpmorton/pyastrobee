"""Quick script to test out modeling with mujoco

This will open the model, begin the simulation, store the video frames, and then play them back

Notes:
- If you want to just load the mjcf in the interactive viewer, run:
    python -m mujoco.viewer --mjcf=PATH_TO_MJCF_FILE
"""

import mujoco
import cv2


def main():
    # Make model and data
    model = mujoco.MjModel.from_xml_path(
        "astrobee_pybullet/mujoco/xml/astrobee_demo.mjcf"
    )
    data = mujoco.MjData(model)

    # Make renderer, render and show the pixels
    renderer = mujoco.Renderer(model)

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)

    # Simulation parameters
    duration = 4  # (seconds)
    framerate = 60  # (Hz)

    # Simulate and display video.
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render().copy()
            frames.append(pixels)
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1000 // framerate) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
