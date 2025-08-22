template_crop_v1 - Here each signal is bandpassed along with being time-cropped (padding outside annotated time interval), and the regular evaluation pipeline is run. The scores were significantly improved but this is not a realistic scenario.

template_crop_v2 - Time-frequency cropped (i.e. bandpassed) prototypes were used to detect in the original clips (only centering, padding to extend beyond source recording bounds). Scores were quite low.

template_crop_v3 - Bandfiltered prototypes are used to detect in bandfiltered clips. Also adds audio normalization.

template_crop_v4 - Same as v3 except for 5s clips.