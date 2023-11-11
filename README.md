<p align="center">
  <img width="460" height="300" src="frontend/assets/icons/icon.svg">
</p>

# JeFaPaTo - A tool for the analysis of facial features

Welcome to **JeFaPaTo**, the Jena Facial Palsy Tool! This powerful tool is designed to assist in various medical and psychological applications by providing accurate facial feature extraction and analysis. It combines the requirements of a medical environment with the possibilities of modern computer vision and machine learning.

Our goal is to allow medical professionals to use state-of-the-art technology without needing to write custom algorithms. We provide the libraries and an interface to the commonly used `mediapipe` library of Google, a powerful tool for facial landmark extraction, and now even offers the distinction into facial movements.

Additionally, our software can be extended to include new methods and algorithms. As initial motivation, we added the **EAR-Score**** (Eye-Aspect-Ratio) to detect blinking and eye closure. This feature is used to analyze the blinking behavior of patients with facial palsy.

## Why use JeFaPaTo?

- **See what you get**: JeFaPaTo offers a real-time preview of the facial landmarks and blendshapes, allowing you to see the results of your analysis as they happen.
- **Feature Extraction**: JeFaPaTo leverages the `mediapipe` library for detailed analysis and tracking of 468 facial landmarks and 52 blend shapes, ideal for medical and psychological experimentation.
- **Easy Feature Selection**: JeFaPaTo allows you to easily select specific facial landmarks and blend shapes for analysis, enhancing flexibility and control in your research or medical investigation. Focus on what you need!
- **Seamless Performance**: Optimized for standard CPUs, JeFaPaTo can process up to 60 FPS for smooth, real-time analysis, offering efficiency and eliminating hardware concerns.
- **Automatic Blinking Detection**: JeFaPaTo's standout feature is its automatic blinking detection, using the Eye Aspect Ratio (EAR) score to simplify identifying blinking patterns for research or diagnosis analysis.
- **Anywhere**: JeFaPaTo is a cross-platform tool that allows you to use it on Windows, Linux, and MacOS.

## Get Started

Ready to dive into the world of precise facial feature extraction and analysis? Give JeFaPaTo a try and experience the power of this tool for yourself! Download the latest version of JeFaPaTo for your operating system from the [releases page](todo) or the following links:

| Windows | Linux | MacOS |
| :-----: | :---: | :---: |

## How to use JeFaPaTo

1. Start JeFaPaTo
2. Select the video file or drag and drop it into the indicated area
3. The face should be found automatically; if not, adjust the bounding box
4. Select the facial features you want to analyze in the sidebar
5. Press the play button to start the analysis

## Citing JeFaPaTo

If you use JeFaPaTo in your research, please cite it as follows:

```bibtex
unpublished yet
```

And depending on the features you use, please also cite the following papers:

```bibtex
mediapipe
```

## Contributing

We are happy to receive contributions from the community. If you want to contribute, please read our [contribution guidelines](CONTRIBUTING.md) first.

## License

JeFaPaTo is licensed under the [MIT License](LICENSE).

## Acknowledgements

JeFaPaTo is based on the [mediapipe]() library by Google. We would like to thank the developers for their great work and the possibility to use their library. Additionally, we would like to thank the [OpenCV](https://opencv.org/) team for their great work and the possibility to use their library. Also, we thank our medical partners for their support and feedback.
