<p align="center">
  <img width="460" height="300" src="frontend/assets/icons/icon.svg">
</p>

# JeFaPaTo - A tool for the analysis of facial features

Welcome to **JeFaPaTo**, the Jena Facial Palsy Tool! This powerful tool is designed to assist in various medical and psychological applications by providing accurate facial feature extraction and analysis. It combines the requirements of a medical environment with the possibilities of modern computer vision and machine learning.

Our goal is to allow medical professionals to use state-of-the-art technology without needing to write custom algorithms. We provide the libraries and an interface to the commonly used `mediapipe` library of Google, a powerful tool for facial landmark extraction, and now even offers the distinction into facial movements.

Additionally, our software can be extended to include new methods and algorithms. We are interested in human blinking behavior and scrutinize it with high temporal videos using the **EAR-Score** (Eye-Aspect-Ratio) to detect blinking and eye closure. This feature is used to analyze patients with facial palsy blinking behavior.

## Why use JeFaPaTo?

- **See what you get**: JeFaPaTo offers a real-time preview of the facial landmarks and blendshapes, allowing you to see the results of your analysis as they happen.
- **Feature Extraction**: JeFaPaTo leverages the `mediapipe` library for detailed analysis and tracking of 468 facial landmarks and 52 blend shapes, ideal for medical and psychological experimentation.
- **Easy Feature Selection**: JeFaPaTo allows you to easily select specific facial landmarks and blend shapes for analysis, enhancing flexibility and control in your research or medical investigation. Focus on what you need!
- **Seamless Performance**: Optimized for standard CPUs, JeFaPaTo can process up to 60 FPS for smooth, real-time analysis, offering efficiency and eliminating hardware concerns.
- **Automatic Blinking Detection**: JeFaPaTo's standout feature is its automatic blinking detection, using the Eye Aspect Ratio (EAR) score to simplify identifying blinking patterns for research or diagnosis analysis. We also give labeling capabilities for individual blinks and a detailed summary of the blink behavior.
- **Support For High Temporal Videos**: The human blink is fast, and JeFaPaTo is designed to handle it. JeFaPaTo can process videos with any FPS but with an extraction optimized for 240 FPS.
- **Anywhere**: JeFaPaTo is a cross-platform tool that allows you to use it on Windows, Linux, and MacOS.

## Get Started

Ready to dive into the world of precise facial feature extraction and analysis? Give JeFaPaTo a try and experience the power of this tool for yourself! Download the latest version of JeFaPaTo for your operating system from the [releases page](https://github.com/cvjena/JeFaPaTo/releases/tag/v1.0.0) or the following links:

- [Windows 11](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_windows.exe)
- [Linux/Ubuntu 22.04](https://github.com/cvjena/JeFaPaTo/releases/latest//download/JeFaPaTo_linux)
- [MacOS Universal2 v13+](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_universal2.dmg)
- [MacOS Intel v13+](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_intel.dmg)
- [MacOS Intel v10.15+](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_intel_v10.dmg)

If you want to install JeFaPaTo from source, please follow the instructions in the [installation guide](INSTALL.md).

## How to use JeFaPaTo

### Facial Features

1. Start JeFaPaTo
2. Select the video file or drag and drop it into the indicated area
3. The face should be found automatically; if not, adjust the bounding box
4. Select the facial features you want to analyze in the sidebar
5. Press the play button to start the analysis

### Blinking Detection

1. Start JeFaPaTo
2. Select the feature "Blinking Detection" in the top bar
3. Drag and drop the `.csv` file containing the EAR-Score values into the indicated area
   - you can also drag and drop the video file into the indicated area to jump to the corresponding frame
4. Press the `Extract Blinks` buttons to extract the blinks (in a future version, the settings are not needed anymore)
5. In the table, you now have the option to label the blinks
6. Press `Summarize` to get a summary of the blink behavior
7. Press `Export` to export the data in the appropriate format

## Citing JeFaPaTo

If you use `JeFaPaTo` in your research, please cite it as follows:

```bibtex
unpublished yet
```

Depending on the features you use, please also cite the following papers:

```bibtex
@article{kartynnikRealtimeFacialSurface2019a,
  title = {Real-Time {{Facial Surface Geometry}} from {{Monocular Video}} on {{Mobile GPUs}}},
  author = {Kartynnik, Yury and Ablavatski, Artsiom and Grishchenko, Ivan and Grundmann, Matthias},
  year = {2019},
  month = jul,
  journal = {ArXiv},
  volume = {abs/1907.06724},
  eprint = {1907.06724},
  primaryclass = {cs},
  doi = {10.48550/arXiv.1907.06724},
}
```

## Contributing

We are happy to receive contributions from the community. If you want to contribute, please read our [contribution guidelines](CONTRIBUTING.md) first.

## License

JeFaPaTo is licensed under the [MIT License](LICENSE.txt).

## Acknowledgements

JeFaPaTo is based on the [mediapipe](https://github.com/google/mediapipe) library by Google. We would like to thank the developers for their great work and the possibility to use their library. Additionally, we would like to thank the [OpenCV](https://opencv.org/) team for their great work and the possibility to use their library. Also, we thank our medical partners for their support and feedback.
