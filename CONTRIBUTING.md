# Contributing to JeFaPaTo

We are happy to receive contributions from the community. If you want to contribute, please read our [contribution guidelines](CONTRIBUTING.md) first (this file :^) )

As JeFaPaTo is a research project, we are not able to pay you for your contributions. However, we will mention you in the [acknowledgements](#acknowledgements) and you will contribute to the scientific community and many researchers.

One unique feature of JeFaPaTo is that it is split into two main parts. The first being the coding library, which is the main area of functionality and the second being the GUI, which is the main area of user interaction. This allows us to have a clear separation of concerns.
If you want to contribute a new feature, please consider which part of the project it belongs to. If you are unsure, please open an issue and we will help you.

## Coding Library

The coding library is the main area of functionality. It contains all the code that is needed to extract the facial features and blink detection. This part should be clearly separated from the GUI and should not contain any GUI code. Basically, this part should be usable without the GUI.

Each feature should be implemented in a separate file. For example, the blink detection is implemented in the file `blink_detection.py`. If you want to contribute a new feature, please create a new file and implement the feature there. If you want to contribute to an existing feature, please open an issue and we will help you.

Also your implementation should be well documented and tested. We use the [Google Style Guide](https://google.github.io/styleguide/pyguide.html) for our code. And please also add tests for your code and if necessary test files. We use [pytest](https://docs.pytest.org/en/6.2.x/) for our tests. We hope to achieve a coverage of `100%` in the coding library, and are happy to receive your help.

## GUI

The GUI is the main area of user interaction. It contains all the code that is needed to interact with the user. This part should be clearly separated from the coding library and should not contain any code that is not related to the GUI.

Here we are happy to receive contributions to improve the user experience. If you have any ideas, please open an issue and we will help you.

It is important to note that this part interacts with the coding library. We include a lot of user sanity input checks in the GUI, but we also need to include them in the coding library. If you are unsure, please open an issue and we will help you.

If you implemented a new feature in the coding library, but have no experience with GUIs, please open an issue and we will help you to bring your feature into the GUI :).

We use `PyQt6` for our GUI to ensure cross-platform compatibility and `pyqtgraph` for our plots. The GUI is build up from several custom widgets to ensure reusability. If you want to contribute a new widget, please create a new file and implement the widget there. If you want to contribute to an existing widget, please open an issue and we will help you.

The GUI is currently not automatically tested. Hence, we still unfortunately have some bugs and rely on feedback of our users. If you find a bug, please open an issue and we will investigate it.

## Acknowledgements

* Oliver Mothes: Initial draft and concept of the project, also the name giver of the project
* Lukas Schuhmann: Medical partner and feedback provider for the windows version
* Elisa Furche: Medical partner and feedback provider for the Mac v10 version 
* Elisabeth Hentschel: Medical partner and feedback provider for the Mac v13 version
* Yuxuan Xie: Investigation of EAR score and blink detection