# Changelog

## 0.1.0 (2026-10-08)


### Features

* campaign store and notes folder loader ([f056af7](https://github.com/RogerHerazo/chronicler/commit/f056af79f5ad08994aae2f302b4e2c26880514e6))
* Claude and Ollama analysis providers ([e1a0162](https://github.com/RogerHerazo/chronicler/commit/e1a016273a0c552de709edab11cdfc192be1d40e))
* command line with serve, doctor and replay ([494e23b](https://github.com/RogerHerazo/chronicler/commit/494e23bd4789a775a5162d1b4711002a0b39bb3c))
* health checks (doctor) ([8996065](https://github.com/RogerHerazo/chronicler/commit/8996065338669d1a9353675dda3eadacbf1af255))
* local transcription with faster-whisper ([4d1b8df](https://github.com/RogerHerazo/chronicler/commit/4d1b8df1d57ee4fdbbdec11184cf3d926d0180bb))
* local web UI ([6d65c3f](https://github.com/RogerHerazo/chronicler/commit/6d65c3f8a150e122b65bd6dc74518aa28b216e02))
* loopback + mic capture, file replay and silence-aware chunking ([7d66250](https://github.com/RogerHerazo/chronicler/commit/7d66250af64f6bc9e65a52178394418c4db78c16))
* session pipeline, campaign tracker updates and Markdown exports ([2e2c76d](https://github.com/RogerHerazo/chronicler/commit/2e2c76dec4965b0ab1cfb13a021af53d060d7c20))
* **ui:** colored level meters and a live level check in Settings ([c083d26](https://github.com/RogerHerazo/chronicler/commit/c083d26cb7cd03d27c9bf61480ade6a00c2af053))


### Bug Fixes

* **audio:** fall back to PortAudio for mics soundcard cannot open ([6500a89](https://github.com/RogerHerazo/chronicler/commit/6500a896e8a9d87ed1d07eecf5e1e07cbae39f70))
* **audio:** mix with headroom and warn when an input clips ([2400710](https://github.com/RogerHerazo/chronicler/commit/24007107c3831ff98d3769a5c164d7af01fe0e0d))
* load CUDA libraries from pip wheels and verify them in the health check ([38e5599](https://github.com/RogerHerazo/chronicler/commit/38e559975e76038eaa2276e8d9bb02d93b2a034c))
* **tracker:** treat entities from the campaign notes as known ([fd428ad](https://github.com/RogerHerazo/chronicler/commit/fd428adf90e9f610d12c69fc705b2a8f9a104436))
* **ui:** polish layout from browser review ([3a65beb](https://github.com/RogerHerazo/chronicler/commit/3a65beb4689cbfa49cdd387ec9e76805037eaf27))
* **ui:** say "1 minute" instead of "1 minutes"; document installing from GitHub ([52cbc2a](https://github.com/RogerHerazo/chronicler/commit/52cbc2abf19a617398847f0595b3217317b695c9))


### Documentation

* explain the level meters and what to do about a clipping mic ([2eee7b6](https://github.com/RogerHerazo/chronicler/commit/2eee7b65c68c1d968dc53c798f9434d49a77eec9))
* README, contributing guide, architecture notes and GitHub setup ([1c462fa](https://github.com/RogerHerazo/chronicler/commit/1c462fafeedb85d10d15a8e93656fce97489c5d9))
