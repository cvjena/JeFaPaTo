# JeFaPaTo

*State: Alpha*

JeFaPaTo is a tool created during the cooperation of the Computer Vision Group Jena (CVG) and the Ear-Nose-Throat Department (ENT) of the University Hospital Jena.
It serves to quickly try out newly developed methods in our group to be tested by doctoral ENT candidates.

The tool is designed to be robust and fast, whereas these requirements depend on the developed method.
Any new addition to the tool requires specific measures and rules.
This approach ensures the correct execution inside the tool and suitable communication with the user.
Thus, correct usage of software development, UI, and UX are keys in the continuous development of JeFaPaTo.

Currently, JeFaPaTo supports the following methods, whereas we distinct between **extraction** and **analysis**.

Extraction (of features):
- Facial Landmark Extraction [dlib, mediapipe]
  - EAR Feature [dlib, mediapipe]

Analysis (of features):
- Blinking analysis based on EAR Feature

## Implementation of a new method

New ideas and methods come up most often in joined meetings between CVG and ENT.
These ideas need to be strictly defined to allow flawless implementation and correct transfer of data *into* and *out of* any written tool.
The ENT and CVG have a regular subgroup meeting to ensure a *Forschungsdatenmanagement Plan*.
These rules should be kept in mind and followed while planning the implementation of a new method.

There are two steps: *joined definitions* and *internal definitions*.

### Joined definitions:

- Data input format (commonly: 2D/3D images, 2D/3D videos, sound files, x-rays)
  - this is most often the data we would receive from the ENT department
  - we have to know which kind of data we can expect from them
  - it has to be made sure if we have to extract features or only do an analysis
  - more specific: file naming convention
- Data output format (tables, graphs, images)
  - we have to decide how the data looks like
  - especially they should define how the data should look, and we build it like that
  - often the naming and which data around the actual data should be specified
  - for the naming of the files, we normally do:
    - <input_name>\_<data_time:YYYY-MM-DD_HH-SS>\_<extra_modifiers>

### Internal definitions:

- this is about the specific internal methods we use
- this can include which kind of backend generates the data
- this should also be somehow be added to the output data to ensure the results can be understood

### Development methods

This section summarizes the information about how to implement a new method inside the existing code structure of JeFaPaTo.

#### Module creation

Todo
#### Logging

### UI

This section specifies how a general UI should look in JeFaPaTo.
Also, the naming of Buttons, Boxes, Labels, and more should be similar to ensure maintainability and a better experience for the user.
Also, we define the colors used for plotting and inside the GUI.

Todo

### UX

The rules for UX include how to notify the user if everything was ok, something failed, some computation failed.

Todo

## Open issues and ideas
- License
  - JeFaPaTo is currently not licensed, and we should add one
  - GPL-v2/v3
- Better looking assets
- Start-up tasks
  - Currently, JeFaPaTo downloads files at the beginning of the start-up if these do not exist
  - We do not handle a missing internet connection
  - There should be a simple way of introducing some automatic start-up checks to reduce the manual calling of the required functions
- Implementation of other landmarking distance methods
  - we have the outline of other ideas for landmarking distance measurements
  - these are not implemented and should be more generalized to achieve a nice GUI and method
  - this requires contact with the ENT
- Tests
  - We do not have any tests, and that should be avoidable
- Add some more project management
  - automatic documentation creation (sphinx?)
  - automatic setup of the development environment of other users?
  - install tox and create several commands like
    - tox test
    - tox export [Windows, Linux, MacOS]
  - we should spend some time and create the building process so we don't have to give them the source code
    - nuitka could be interesting, or
    - pyinstaller
  - we should slowly get rid of the installation of packages inside conda (apart from python). This change would improve the speed more and is more stable
  - improve exportation to different systems (ships with python itself?)

## Export and installation of JeFaPaTo

Currently, we export the whole project for a specific OS and provide some start-up scripts for the user.
This process is slow and easily disturbed.
The users should be bothered with coding, changing files, or others.

We use ```export.py``` to create a ZIP file for the project, then send it to the users.

### Linux

Todo

### Windows

Todo

### MacOS

Todo
