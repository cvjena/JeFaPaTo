---
title: 'JeFaPaTo - A joint toolbox for blinking analysis and facial features extraction'
tags:
  - Python
  - Blinking
  - Facial Analysis 
  - Blend Shapes
  - Facial Expressions
authors:
  - name: Tim Büchner
    orcid: 0000-0002-6879-552X
    corresponding: true 
    affiliation: 1
  - name: Oliver Mothes
    orcid: 0000-0002-2294-3670
    affiliation: 1
  - name: Gerd Fabian Volk
    orcid: 0000-0003-1245-6331
    affiliation: 2
  - name: Orlando Guntinas-Lichius
    orcid: 0000-0001-9671-0784
    affiliation: 2
  - name: Joachim Denzler
    orcid: 0000-0002-3193-3300
    affiliation: 1
affiliations:
 - name: Computer Vision Group, Friedrich Schiller University Jena, 07743 Jena, Germany
   index: 1
 - name: Department of Otorhinolaryngology, Jena University Hospital, 07747 Jena, Germany
   index: 2
date: 17 November 2023
bibliography: paper.bib

---

# Summary

The analysis of facial features and expressions is a challenging task in computer vision.
The human face is a complex object with a high degree of variability in shape, texture, and appearance.
Especially for medical areas deviating from normal facial structures, e.g. due to paralysis, are of interest and require a precise analysis.
The delicate movement of the eye blink is a yet to be fully understood process and requires high temporal resolution for detailed analysis.
However, many modern computer vision approaches require programming skills to be used and are not easily integrable into the workflow of medical experts.
`JeFaPaTo` - the Jena Facial Palsy Tool - aims to overcome this gap by leveraging modern computer vision algorithms and providing a user-friendly interface for non-programmers.

The state of the eye is of high interest for medical experts, e.g. in the context of facial palsy or Parkinson's disease.
Due to facial nerve damage, the eye closing process might be impaired and could lead to many undesirable side effects.
Hence, a simple distinction between open and closed eyes is not sufficient for a detailed analysis.
Factors such as duration, synchroncity, velocity, full closure, the time between blinks, and blink frequency over time are of high interest.
Such detailed analysis could help medical experts to better understand the blinking process, its deviations, and possible treatments for better eye health.

# Statement of need

To analyze the blinking behavior in detail, medical experts often use high-speed cameras to record the blinking process.
Therefore, the video data is often recorded at 240 FPS or higher, which results in a large amounts of data, and require optimized algorithms for consumer hardware.
`JeFaPaTo` is a Python-based program to support medical and psychological experts in the analysis of blinking and facial features for high temporal resolution video data.
The tool is divided into two main parts: An extendable programming interface and a graphical user interface (GUI) fully written in Python.
The programming interface reduces the overhead of dealing with high temporal resolution video data, automatically extracts selected facial features, and provides a set of analysis functions specialized for blinking analysis.
The GUI provides non-programmers an intuitive way to use the analysis functions, visualize the results, and export the data for further analysis.
`JeFaPaTo` is designed to be extendable by additional analysis functions and facial features, and is under joint development by computer vision and medical experts to ensure a high usability and relevance for the target group.

`JeFaPoTo` leverages the `mediapipe` library [@mediapipe] to extract facial landmarks and blend shape features from video data at 60 FPS (on modern hardware).
The extracted features are used to compute the `EAR` (Eye-Aspect-Ratio) [@ear] for both eyes over the videos.
Additionally, `JeFaPaTo` detects blinks, matches left and right eye, and computes a summary, shown in \autoref{fig:summary}, for the provided video and exports the data in various formats for further independent analysis.
We leverage `PyQt6` [@pyqt6] and `pyqtgraph` [@pyqtgraph] to provide a GUI on any platform for easy usage.

![A visual summary of the blinking behavior during a single 20 minute video recorded at 240 FPS.\label{fig:summary}](img/summary.png)

To support and simplify the usage of `JeFaPaTo`, we provide as a standalone executable for Windows, Linux and MacOS.
`JeFaPaTo` is currently used in three medical studies to analyze the blinking process of healthy probands, and patients with facial palsy and Parkinson's disease.

# Functionality and Usage

`JeFaPaTo` was developed to support medical experts in the extraction and analysis of the blinking behavior.
Hence, the correct localization of facial landmarks is of high importance and the first step in the analysis process of each frame.
Once a user provides a video in the GUI, an automatic face detection is performed and the user can adapt the bounding box if necessary.
Due to the usage of `mediapipe` [@mediapipe] the tool is able to extract 468 facial landmarks and additional 52 blend shape features.
To describe the state of the eye, we use the Eye-Aspect-Ratio (EAR) [@ear], which is a common measure for blinking behavior and is computed based on the 2D coordinates of the landmarks.
As this measure describes the ratio between the vertical and horizontal distance between the landmarks, the detailed behavior of the upper and lower eyelid is captured.
Please note, that all connotations for left and right eye are based on the perspective of the person in the video.

We denote this measure as `EAR-2D-6` and the according six facial landmarks are selected for both eyes, as shown in \autoref{fig:ear}, and is computed for each frame.
As `mediapipe` belongs to the monocular approaches for facial reconstruction, each landmark contains an estimated depth value.
We offer the `EAR-3D-6` feature, which is computed based on the 3D coordinates of the landmarks, to leverage this information to minimize the influence of head rotation.
However, first experiments indicated that the 2D approach is sufficient enough for the analysis of blinking behavior.

![Visualization of the Eye-Aspect-Ratio for the left (blue) and right (red) eye.\label{fig:ear}](img/ear.png)

The `EAR` score is computed for each frame and results in a time series saved to a CSV file.
The resulting file can then be used to detect the blinking behavior, as shown in \autoref{fig:summary}, and compute the summary statistics.
The extraction is based on the `scipy.signal.find_peaks` algorithm [@scipy], and the time-series can be smooth if necessary.
`JeFaPaTo` will automatically match the left and right eye blinks and provide the statistics for both eyes.

After the extraction, the user has the option to manually correct the blinking behavior in a table, e.g. labeling the state of blinking as `none`, `partial`, or `full` closure.
To simplify this process, the user can drag-and-drop the according video into the GUI, and `JeFaPaTo` will jump to the according frame.
Once, the blinking behavior is corrected, the summary statistics are computed and the data can be exported for further analysis.
The data can be exported as a CSV files or as a single Excel file, which contains the time-series of the `EAR` score, the blinking behavior, and the summary statistics.
We provide a sample file for the score in the `examples/` directory of the repository.

# Platform Support

As `JeFaPaTo` is written in Python, it can be used on any platform that supports Python.
The tool can be run by cloning the repository, using the `dev_init.sh` script to create the according `conda` environment with all dependencies.
As `JeFaPaTo` is intended to be used by non-programmers, we support most common platforms.
We provide with each release a standalone executable for `Windows 10`, `Linux (Ubuntu 22.04)`, and `MacOS (version 13+ for both Apple Silicon and Intel)`.
We offer a separate branch for `MacOS version 10+ (Intel)`, which does not contain blend shape extraction, to support older hardware.
All user interface and experience tests are conducted on `Windows 10` and `MacOs 13+ (Apple Silicon)` by the authors and medical partners.

# Ongoing Development

`JeFaPaTo` finished the first stable release and will continue to be developed to support the analysis of facial features and expressions.
As high temporal resolution video data might open up new insights into facial movements, we plan to implement common 2D measurement based features.
Additionally, a common side effect of facial palsy are synkinesis, which is the involuntary movement of facial muscles.
It is often observed that the eye closes when the patient smiles, or the mouth moves when the patient closes the eye.
Hence a joint analysis of the blinking pattern and mouth movement could help to better understand the underlying processes.
The EAR is sensitive to head rotation but can be largely avoided due to the experimental setup.
To support the analysis of facial palsy patients, we plan to implement a 3D head pose estimation to correct the EAR score for head rotation in the future.

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

# Acknowledgements

Supported by Deutsche Forschungsgemeinschaft (DFG - German Research Foundation) project 427899908 BRIDGING THE GAP: MIMICS AND MUSCLES (DE 735/15-1 and GU 463/12-1).
We acknowledge the helpful feedback for the user-interface development and quality-of-life requests from Lukas Schuhmann, Elisa Furche, Elisabeth Hentschel, and Yuxuan Xie.

# References