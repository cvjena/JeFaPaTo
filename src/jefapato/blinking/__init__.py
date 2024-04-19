__all__ = ["peaks", "smooth", "match", "summarize", "visualize", "load_ear_score", "save_results", "HELP_DESCRIPTION"]

from jefapato.blinking.peaks import peaks
from jefapato.blinking.smooth import smooth
from jefapato.blinking.match import match
from jefapato.blinking.summary import summarize, visualize
from jefapato.blinking.io import load_ear_score, save_results


HELP_DESCRIPTION = """
<h2>Recommended Extracton Settings for Eye Blinking</h2>

<p>
Here we give some recommended settings for the extraction of eye blinking from EAR scores.
These settings are based on the analysis of the EAR scores of a dataset of internal dataset, not yet published.
</p>

<p>
The dataset was recorded at 30 and 240 FPS.
The settings are divided into two groups, one for each FPS.
These should be a good starting point for the extraction of eye blinking from EAR scores depending on the FPS of the video.
</p>

<p>
The settings are the following:
</p>
<hr>

<p>@30 FPS</p>
<ul>
    <li>Minimum Distance: 10 Frames</li>
    <li>Minimum Prominence: 0.1 EAR Score</li>
    <li>Minimum Internal Width: 4 Frames</li>
    <li>Maximum Internal Width: 20 Frames</li>
    <li>Maximum Matching Distance: 15 Frames</li>
    <li>Partial Threshold Left: 0.18 EAR Score</li>
    <li>Partial Threshold Right: 0.18 EAR Score</li>
</ul>
<p>Smoothing</p>
<ul>
    <li>Window Size: 7</li>
    <li>Polynomial Degree: 3</li>
</ul>
<hr>

<p>@240 FPS</p>
<ul>
    <li>Minimum Distance: 50 Frames</li>
    <li>Minimum Prominence: 0.1 EAR Score</li>
    <li>Minimum Internal Width: 20 Frames</li>
    <li>Maximum Internal Width: 100 Frames</li>
    <li>Maximum Matching Distance: 30 Frames</li>
    <li>Partial Threshold Left: 0.18 EAR Score</li>
    <li>Partial Threshold Right: 0.18 EAR Score</li>
</ul>
<p>Smoothing</p>
<ul>
    <li>Window Size: 7</li>
    <li>Polynomial Degree: 3</li>
</ul>
"""