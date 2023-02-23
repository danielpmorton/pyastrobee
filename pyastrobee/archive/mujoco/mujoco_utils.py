"""Helper functions for Mujoco"""

import mujoco


def convert_urdf_to_mjcf(urdf_filepath: str, output_filepath: str):
    """Converts a URDF file to MJCF format

    Alternatively, you could load the urdf in the interactive viewer GUI and
    click on the "Save XML" button from there

    Args:
        urdf_filepath (str): Path to the URDF file to be converted
        output_filepath (str): Path to where the MJCF file should be saved
    """
    model = mujoco.MjModel.from_xml_path(urdf_filepath)
    mujoco.mj_saveLastXML(output_filepath, model)
