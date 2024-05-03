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

## Getting Started

Ready to dive into the world of precise facial feature extraction and analysis? Give JeFaPaTo a try and experience the power of this tool for yourself! Download the latest version of JeFaPaTo for your operating system from the [releases page](https://github.com/cvjena/JeFaPaTo/releases) or the following links:

- [Windows 11](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_windows.exe)
- [Linux/Ubuntu 22.04](https://github.com/cvjena/JeFaPaTo/releases/latest//download/JeFaPaTo_linux)
- [MacOS Universal2 v13+](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_universal2.dmg)
- [MacOS Intel v13+](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_intel.dmg)
- [MacOS Intel v10.15+](https://github.com/cvjena/JeFaPaTo/releases/latest/download/JeFaPaTo_intel_v10.dmg)

## Tutorials

If you want to know more about how to use `JeFaPaTo`, please refer to the [Wiki Pages](https://github.com/cvjena/JeFaPaTo/wiki).
There, you can find a custom installation guide and two tutorials, one for the facial feature extraction and another one for the eye blink extraction.
Additionally, we list specific background information on the usage of the tool.

## Citing JeFaPaTo

If you use `JeFaPaTo` in your research, please cite it as follows:

```bibtex
@article{Büchner2024,
  doi = {10.21105/joss.06425},
  url = {https://doi.org/10.21105/joss.06425},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {97},
  pages = {6425},
  author = {Tim Büchner and Oliver Mothes and Orlando Guntinas-Lichius and Joachim Denzler},
  title = {JeFaPaTo - A joint toolbox for blinking analysis and facial features extraction},
  journal = {Journal of Open Source Software}
  }
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
