# PushT successes with visible block movement

All three saved action traces were replayed successfully. Videos show actual observations and decoded predictions using the recorded scheduler-selected fidelity, with five-action prediction endpoints. Selection prioritizes block movement and reasonably long trajectories.

| Episode | Actions | Block net displacement | Maximum rotation from start | Video |
|---|---:|---:|---:|---|
| 40 | 39 | 135.1 | 25.5° | [MP4](pusht_episode_040_39_steps_scheduler.mp4) |
| 6 | 50 | 105.8 | 36.8° | [MP4](pusht_episode_006_50_steps_scheduler.mp4) |
| 28 | 40 | 68.8 | 56.1° | [MP4](pusht_episode_028_40_steps_scheduler.mp4) |

Distances are simulator coordinate units on the 512 × 512 workspace. Rotation is the maximum absolute wrapped angle difference from the initial block angle. These are measured from actual replayed physics states, not decoded predictions.

The earlier 82-action clip (episode 73, goal offset 50) has zero block motion. The earlier 40-action clip (episode 56) moves only 0.57 units. The earlier 41-action clip (episode 28) does move the block 58.3 units. All satisfy the configured success criterion; the new selection is more useful for demonstrating pushing.

`motion_results.json` contains screened replay measurements and block poses; `screening_results.json` contains the three selected examples. Per-episode directories contain scheduler provenance and cached frames.
