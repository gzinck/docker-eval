# Image Processing with Docker Checklist

This checklist serves as a guide for researchers who are interested in incorporating Docker into their image processing workflows and experiments.

## Docker
- [ ] even when Docker is used, underlying differences in hardware can affect performance (e.g., systems with more cores and RAM will be faster no matter what), so system information should still be disclosed when Docker is used for experiments
- [ ] resource allocations in the Docker Desktop settings (e.g., number of CPUs and memory allocated to Docker) should be disclosed to help with reproducibility
    - [ ] some versions of the Docker Desktop app do not allow users to customize Docker resource allocations (e.g., some Windows installations), so this should be disclosed as applicable

## Dockerfiles

For all Dockerfiles, check that they include:
- [ ] version numbers for all installed packages
- [ ] references to public Git repositories only
- [ ] comments to provide clarification
- [ ] only use the minimal set of packages needed to execute the experiment

## Image Processing Benchmarks

When creating or using image processing benchmarks, ensure that:
- [ ] a variety of image resolutions are tested
- [ ] for feature detection benchmarks, there are images that contain those features
- [ ] each benchmark's performance is measurable at the precision level of the machines being tested
