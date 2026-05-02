<div align="center">

# 💡LAMP: Localization Aware Multi-camera People Tracking in Metric 3D World

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://facebookresearch.github.io/LAMP)
[![arXiv](https://img.shields.io/badge/arXiv-2605.05390-b31b1b)](https://arxiv.org/abs/2605.05390)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](https://youtu.be/pJv1xJ-ssUQ)

**CVPR 2026**

[Nan Yang](https://nan-yang.me/) · [Julian Straub](https://jstraub.github.io/) · [Fan Zhang]() · [Richard Newcombe](https://rapiderobot.bitbucket.io/) · [Jakob Engel](https://jakobengel.github.io/) · [Lingni Ma](https://scholar.google.com/citations?user=eUAgpwkAAAAJ&hl=en)

*Meta Reality Labs Research*

</div>

![LAMP teaser](resources/imgs/lamp_teaser.png)

LAMP tracks 3D human motion from egocentric multi-camera headsets via early disentanglement of observer and target motion. Using known device 6-DoF motion and calibration, 2D body keypoints from all cameras over a temporal window are lifted into a unified 3D world reference frame, and an end-to-end trained spatio-temporal transformer fits 3D human motion directly to this 3D ray cloud. This "lift-then-fit" approach achieves state-of-the-art results on monocular benchmarks while significantly outperforming baselines on the targeted egocentric setting.

## News

- **Code release coming soon — please stay tuned!**

## Citation

```bibtex
@inproceedings{yang2026lamp,
  title     = {{LAMP}: Localization Aware Multi-camera People Tracking in Metric {3D} World},
  author    = {Yang, Nan and Straub, Julian and Zhang, Fan and Newcombe, Richard and Engel, Jakob and Ma, Lingni},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

LAMP is CC-BY-NC licensed, as found in the [LICENSE](LICENSE) file.
