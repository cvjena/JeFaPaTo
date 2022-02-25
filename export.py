import pathlib
import zipfile

import structlog

logger = structlog.get_logger()


def write_dir_to_zip(zfile: zipfile.ZipFile, dir_path: pathlib.Path) -> None:
    """
    Write a directory to a zip file.

    :param zfile: Zip file to write to.
    :param dir_path: Path to directory to write.
    """
    logger.info("Writing directory to zip file", dir_path=dir_path)

    for child in dir_path.iterdir():
        if child.is_file():
            logger.info("Writing file to zip file", file_path=child)
            zfile.write(child.as_posix())
        elif child.is_dir():
            if child.name == "__pycache__":
                continue
            zfile.write(child.as_posix())
            write_dir_to_zip(zfile, child)


version = "2022.02.25"
file_name = f"JeFaPaTo-{version}-Alpha"
export_dir = pathlib.Path("__exports__")
export_dir.mkdir(exist_ok=True, parents=True)
file_dir = export_dir / f"{file_name}.zip"
jefapato_z = zipfile.ZipFile(file_dir, "w")
system = "Windows"

logger.info(
    "Start export of JeFaPaTo",
    version=version,
    file_name=file_name,
    system=system,
    file_dir=file_dir,
)
logger.info("Writing files to zip file", file_name=file_name)

# add assets
write_dir_to_zip(jefapato_z, pathlib.Path("assets"))
# add frontend
write_dir_to_zip(jefapato_z, pathlib.Path("frontend"))
# add jefapato
write_dir_to_zip(jefapato_z, pathlib.Path("jefapato"))
# add ui
write_dir_to_zip(jefapato_z, pathlib.Path("ui"))

# add environment
jefapato_z.write("env.yml")
logger.info("Writing environment to zip file", file_name=file_name)

# add main file
jefapato_z.write("main.py")
logger.info("Write main file to zip file", file_name=file_name)

# add bash scripts
jefapato_z.write("jefapato_create.bat")
jefapato_z.write("jefapato_start.bat")
jefapato_z.write("jefapato_update.bat")
logger.info("Write bash scripts to zip file", file_name=file_name)

# add README and LICENSE
jefapato_z.write("LICENSE.txt")
# jefapato_z.write("README.md")
logger.info("Write LICENSE to zip file", file_name=file_name)

jefapato_z.close()
logger.info("Closed zip file", file_name=file_name)
logger.info("Export complete", file_name=file_name)
