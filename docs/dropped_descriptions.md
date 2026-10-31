# Figures MinerU left undescribed, and what DeepSeek-OCR-2 wrote for them

Numbers from `scripts/dropped_descriptions.py`. Score splits use the same paper bootstrap and equal split
weighting as `scripts/analyze.py`.

Setups are named as in the README, shortened: text only = MinerU → V4 Flash, images = MinerU + figure images
→ V4 Flash Vision, descriptions = MinerU + figure descriptions → V4 Flash, OCR-2 = DeepSeek-OCR-2 → V4 Flash.

## Short version

- MinerU's advanced tier left 356 of 1,512 figures (24%) without a description. No analysis call failed.
  MinerU analyses photos and diagrams as well as charts, but it never analyses a figure its layout pass sees
  as several images (a multi-image container), or anything inside one, and it flattens those to a plain image
  afterwards. That accounts for 315 skips; 20 were below its size cutoff and 21 it never boxed. A MinerU
  maintainer confirms the multi-image exclusion is deliberate (issue #5606); the docs don't mention it.
- OCR-2 read 179 of those 315 figures as one image and 136 as several separate crops.
- The skipped figures are mostly plots and photos, and multi-panel figures are skipped far more often
  (42% against 17%).
- OCR-2 wrote something for 343 of those 356 figures. In a hand check of 10 of them, 7 readings were wrong
  or meaningless, 1 was partly right and 2 were roughly right. The common failures are prose that invents
  what the figure shows, and charts or diagrams forced into tables of zeros or swapped series.
- On the 114 questions whose figures MinerU skipped, nothing helped: not the images, not OCR-2's readings.
  On the 367 questions whose figures it did describe, all three ways of adding figure content gained about
  4 points. The gap between the two groups is outside the noise for all three setups.
- So MinerU's skips cost the descriptions setup little. They fall on figures that the other two methods couldn't use either.

## 1. What MinerU skipped, and why

Each figure here is one of the standard-tier figures that the text-only, images and descriptions setups use
(panels regrouped under their "Figure N" caption), matched to the advanced run exactly as the descriptions
setup does.

| | Figures | Share |
| --- | --- | --- |
| Described | 1,156 | 76% |
| Grouped into a multi-image container, then flattened to a plain image | 315 | 21% |
| Below MinerU's size cutoff | 20 | 1% |
| No box in the advanced run at all | 21 | 1% |
| Total | 1,512 | |

### Why: how MinerU 4.0.12 chooses what to analyse

Image analysis isn't limited to charts. MinerU sends both `image` and `chart` blocks to it
(`IMAGE_ANALYSIS_TYPES = {"image", "chart"}` in `mineru_vl_utils/mineru_client.py`), and the advanced run did
describe 300+ photos (`natural_image`), 400+ diagrams (`flowchart`), text images and chemical structures. What
decides it is layout, in `MinerUClientHelper.prepare_for_extract`:

1. **Multi-image containers are never sent.** When MinerU's layout pass draws one `image_block` box around
   several images or charts (a figure with panels, a grid of photos), the container is on the skip list
   (`skip_list = {"list", "equation_block", "image_block"}`).
2. **Nor is anything inside one.** Any image or chart at least 90% inside a container counts as
   "non-standalone" and is skipped too. So neither the whole figure nor its panels get analysed.
3. **Small images are skipped.** A block needs to be more than 10% of the page in both width and height, or
   more than 1% of the page's area (`_is_eligible_for_image_analysis`).
4. **The container is then flattened.** Later, `_collapse_image_blocks` (`docvortex/document/pdf/visuals.py`)
   turns each container into one plain `image` block, with no sub-type and an empty body, and deletes the
   panels inside it. That is the undescribed "plain image" in the output.

There is a second route to step 4 in the advanced tier (effort `xhigh`):
`_supplement_missing_image_block_containers` (`mineru/backend/analysis/pdf/visual_containers.py`) adds a
container wherever MinerU's local layout model sees one figure box holding at least 2 images or charts that
cover at least 90% of it. That runs in `_finish_pdf_window`, after the VLM step, so panels that were already
analysed get their analyses deleted when the container collapses. The saved output is already flattened, so
we can't tell how many of the 315 took this route rather than step 1.

How each skipped figure was classified: a box in the advanced run's output with no sub-type that passes the
size rule can only be a flattened container. There are 369 such blocks across the 300 papers, 315 of them
on figures we use; the rest aren't matched to a standard-run figure. A box that fails the size rule was
skipped for size. Every block with a sub-type has a description, except one line chart.

MinerU's maintainers confirm this is intentional, though only in an issue thread. In
[opendatalab/MinerU#5606](https://github.com/opendatalab/MinerU/issues/5606) (4.0.9, Advanced tier, "some large
figures get no image_analysis result"), maintainer myhloli replied on 2026-10-09, translated: large blocks
merged from several small images hold too much information for the current image analysis, so it doesn't run
on them, on any tier, Advanced included. The issue was closed as completed with no code change. Neither
MinerU's docs (Tiers and Runtimes, output format reference) nor the MinerU2.5 and MinerU2.5-Pro papers
(arXiv 2509.22186, 2604.04771) mention the exclusion. The Pro paper describes image analysis as working on
"cropped image regions" and says image-analysis data hasn't yet had its data-engine treatment.

So the rule isn't "plain images are skipped". It's "any figure MinerU sees as several images is skipped". A
grid of photos is the clearest case: A/1701.03077v10 Fig. 15 is 14 rows of face crops, a single image in the
standard run, and comes out of the advanced run as a flattened container. 154 of the 315 containers were one
block in the standard run. The advanced tier runs an extra local layout model (the second route above), which
may be what finds parts in figures the standard tier kept whole.

Skip rate by figure kind:

| | Skipped | Of | Rate |
| --- | --- | --- | --- |
| Single panel (as the standard run split it) | 184 | 1,103 | 17% |
| Several panels | 172 | 409 | 42% |
| SPIQA type: plot | 143 | 544 | 26% |
| SPIQA type: photo / visualization | 77 | 238 | 32% |
| SPIQA type: schematic | 62 | 475 | 13% |
| SPIQA type: other | 12 | 49 | 24% |
| SPIQA type: table (figure-captioned tables) | 11 | 34 | 32% |
| Not matched to a SPIQA figure | 51 | 172 | 30% |

The panel count comes from the standard run, which splits panels differently from the advanced run, so
"single panel" here still includes some figures the advanced run saw as containers.

Diagrams are skipped least (13%).

Questions see 114 of the skipped figures (134 question-figure references, out of 537).

## 2. What OCR-2 wrote for the same figures

OCR-2's boxes were matched to each MinerU figure by page and overlap (at least half of the smaller box).

| | Skipped by MinerU (356) | Described by MinerU (1,156) |
| --- | --- | --- |
| OCR-2 boxed it as an image and read it | 343 | 1,144 |
| OCR-2 saw only text, table or equation boxes there | 13 | 10 |
| On a page OCR-2 dropped | 0 | 2 |
| OCR-2 reading is an HTML table | 214 (60%) | 773 (67%) |
| OCR-2 reading is prose | 129 | 371 |
| Reading cut off at the token limit | 7 | 48 |
| Tables with at least half their cells empty | 9 | 70 |
| Median length (characters) | 2,314 | 1,767 |
| For comparison, median MinerU description | — | 430 |

OCR-2 almost always produces something, and it is long: its readings run four to five times the length of
MinerU's. "Parse the figure." makes it turn most figures into a data table, whatever the figure is.

### Did OCR-2 read MinerU's multi-image figures whole or in parts?

Both, about evenly. OCR-2 reads each image box its page pass draws as a separate crop, with no knowledge of
the other crops, and `scripts/ocr2_pipeline.py` joins the readings under the one figure. So the number of
OCR-2 image boxes over a figure is the number of independent readings it got.

| MinerU figures | Figures | OCR-2: one box (read whole) | 2–3 boxes | 4+ boxes |
| --- | --- | --- | --- | --- |
| Flattened containers (skipped) | 315 | 179 | 95 | 41 |
| ...of which several panels in the standard run | 161 | 45 | 82 | 34 |
| ...of which one block in the standard run | 154 | 134 | 13 | 7 |
| For comparison: described, several panels | 237 | 50 | 113 | 74 |
| For comparison: described, one block | 919 | 892 | 12 | 3 |

OCR-2 mostly splits where the standard run splits: when it sees separate plots it boxes them separately,
and the photo grids and diagrams MinerU's advanced tier alone treated as multi-image, OCR-2 reads whole. The
two groups read differently:

| Containers OCR-2 read as... | Figures | Reading is a table | Reading is prose | Most common SPIQA types |
| --- | --- | --- | --- | --- |
| One box | 179 | 85 | 94 | photo 52, schematic 52, plot 45 |
| Several boxes | 136 | 117 | 19 | plot 95, photo 16 |

Read whole, a photo grid or diagram usually gets prose, which is where the invented descriptions in the hand
check come from. Read in parts, each plot gets its own table. That is closer to what the figure holds, but each
part is read without the others or the caption, so it can't know which panel is which condition.

### Hand check: 10 skipped figures that questions reference

Each OCR-2 reading compared with the SPIQA figure image.

| Figure | What it shows | What OCR-2 wrote | Verdict |
| --- | --- | --- | --- |
| A/1705.09882v2 Fig. 1 | Grids of CNN filter responses for person re-ID (RGB and depth) | Prose about a "Sparse-GAN" generating images over 1,000 and 2,000 training iterations | Invented |
| A/1706.00827v2 Fig. 4 | Photos with points coloured by fitted plane | Prose about 3D object detection, dots coloured as "door", "window", "roof" | Invented (scene right, meaning wrong) |
| A/1705.09966v2 Fig. 13 | Frontal faces generated from side faces | Prose about face recognition improving from 0 to 1,000,000 training samples | Invented |
| C/1909.06162 Fig. 1 | System diagram: features, classifiers, LSTM-CRF tagger | Picks up real labels (wordLSTM, CRF, B-Slogan, FastText BERT) but explains them as bilingual English-Chinese sentiment | Partly right |
| A/1809.00458v1 Fig. 12 | Two line plots, GB-KMV above LSH-E | Table with the two methods' values swapped, so LSH-E looks better | Wrong on the key fact |
| A/1811.02553v4 Fig. 17 | Grid of 3D reward-landscape surfaces | Table of near-zero numbers | Meaningless |
| A/1706.00633v4 Fig. 1 | Decision-boundary sketch, t-SNE plots, detector diagram | Table of "Decision boundary" rows, all 0.0 | Meaningless |
| B/1502.04681 Fig. 7 | Greyscale LSTM activation maps | Table of zeros under the panel titles | Meaningless |
| B/1512.03385 Fig. 4 | ResNet training curves | Tables of error by iteration; values rough, ordering of the curves right | Roughly right |
| A/1901.00056v2 Fig. 3 | Three small line plots of AUC and MAP | Tables with values close to the plotted ones | Roughly right |

Ten figures is a small sample, chosen at random from the referenced skipped figures (seed 0). It shows the
pattern clearly but doesn't give a reliable error rate.

The prose failures have a common shape: OCR-2 sees only the crop, not the paper, so it guesses a topic and
writes a confident, detailed description of a paper that doesn't exist. The table failures come from figures
with little readable text in them (activation maps, 3D surfaces, sketches): OCR-2 still produces a table, but
its numbers aren't in the figure.

### For comparison: 6 figures MinerU did describe

| Figure | MinerU | OCR-2 |
| --- | --- | --- |
| B/1707.01083 Fig. 2 (ShuffleNet units) | Three mermaid flowcharts, structure right | One table of layers × units with 1s, structure lost |
| A/1802.07459v2 Fig. 2 (pipeline overview) | Mermaid flowchart with the stage names | Prose with right stage names, under an invented paper title |
| A/1805.06431v4 Fig. 3 (ChoiceNet) | Mermaid graph of the blocks | Table of block names with empty values |
| A/1804.01429v3 Fig. 5 (small pipeline) | Short mermaid graph, right | Table repeating "3D ConvNet" |
| A/1811.02721v3 Fig. 13 (photos of a device) | One line per photo: a green circuit board | Long prose calling it a 3D-printed wind-energy anemometer with four arms |
| B/2206.03715 Fig. 6 (two colour heatmaps without numbers) | Tables of approximate values, mostly wrong sign | Tables of 1s and 0.2s, also wrong |

On diagrams, MinerU's mermaid keeps the structure and OCR-2's tables lose it. On photos, MinerU stays short and
literal while OCR-2 embellishes. Neither can read a heatmap that has no numbers printed on it.

## 3. Effect on scores

Questions grouped by whether the figures they reference reached the descriptions setup with a description.
Differences are from text only; brackets are 95% paper-bootstrap ranges.

| Questions reference... | Questions | Text only | Images | Descriptions | OCR-2 |
| --- | --- | --- | --- | --- | --- |
| Only tables | 434 | 61.1 | −0.4 (−2.6 to +2.0) | +1.4 (−0.7 to +3.3) | −1.2 (−3.1 to +0.7) |
| Figures MinerU described | 367 | 65.2 | +4.0 (+0.5 to +7.8) | +3.5 (−0.1 to +7.3) | +3.9 (+0.3 to +7.5) |
| Figures MinerU skipped | 114 | 67.3 | −1.8 (−6.2 to +2.1) | −1.9 (−5.4 to +1.2) | −2.7 (−6.6 to +0.7) |
| Some of each | 5 | 48.4 | +2.1 | +8.1 | +8.3 |
| A figure MinerU never found under its label | 47 | 77.8 | +0.2 (−7.3 to +9.1) | −3.3 (−23.8 to +8.6) | −9.2 (−30.1 to −0.6) |

Gain on described-figure questions minus gain on skipped-figure questions:

| | Gap | 95% range |
| --- | --- | --- |
| Images | 5.8 | +0.6 to +11.3 |
| Descriptions | 5.4 | +0.8 to +10.4 |
| OCR-2 | 6.6 | +1.6 to +12.0 |

What this says:

- If MinerU's skips were what held the descriptions setup back, it should lose on skipped-figure questions
  while the images and OCR-2 setups, which do see those figures, gain there. Instead all three lose a little there and gain on the same described-figure
  questions. The skipped figures are ones where extra figure content didn't help, whichever way it was
  supplied.
- For OCR-2, that matches the hand check: on these figures its readings are often wrong, so having a reading
  isn't the same as having the figure's content. Text only already scores higher on these questions than on
  the others (67.3 against 65.2), so they may lean on the caption and body text more.
- The descriptions setup's gain over text only comes from the described figures and the table questions, not from coverage. OCR-2's
  near-complete coverage (99.96% of boxes) bought nothing on the figures MinerU missed.
- OCR-2's loss on table-only questions (descriptions minus OCR-2 there: +2.6, +0.3 to +4.8) points to its own
  table and text conversion, not its figure readings, as the reason it ends up level with text only overall.

The 367-question group is the first place in this study where adding figure content clears the noise (for images
and OCR-2).

## Caveats

- The described/skipped gap holds for all three setups, but they answered the same questions, so it is one
  observation, not three.
- The groups differ in more than description status: skipped figures are more often plots, photos and
  multi-panel figures. The split can't separate "MinerU skipped it" from "it's a kind of figure that's hard to
  use".
- The hand check is 10 skipped and 6 described figures, judged by one reader against the SPIQA images.
- OCR-2 boxes are matched to MinerU figures by overlap. A few may pick up a neighbouring box or miss one
  that OCR-2 drew much larger or smaller.
- The 47 "never found under its label" questions reference a figure MinerU's captions don't label as that
  figure number. These are mostly label-parsing misses, not missing figures; the group is small and its ranges
  are wide.

## Would describing each panel separately help?

MinerU already boxes the panels inside a container and skips them on purpose, so describing them would be a
small change. We didn't test it directly, but OCR-2 shows what per-panel readings do: of the containers MinerU
skipped, OCR-2 read 136 one panel at a time. Questions about skipped figures, split by how OCR-2 read them;
differences are from text only, net of the same setup's difference on table-only questions, which takes out
OCR-2's general text deficit:

| Questions about containers OCR-2 read... | Questions | Text only | Images, net | OCR-2, net |
| --- | --- | --- | --- | --- |
| In parts, one reading per panel | 43 | 54.2 | −0.3 (−3.9 to +4.1) | −0.9 (−7.3 to +5.6) |
| Whole, one reading | 63 | 67.5 | −1.8 (−9.6 to +4.5) | −2.5 (−7.8 to +2.4) |

Per-panel readings didn't help, and neither did the images. The groups are small, so a gain of a few points
can't be ruled out. As a rough upper bound: if per-panel descriptions added as much on the 114 skipped-figure
questions as MinerU's descriptions added on the figures it did describe (+3.5), the overall score would move
by about 0.4 points, which this test couldn't detect.

Per-panel reading also has a weakness of its own. Each panel is read alone, without the caption or the other
panels, so a reading can't say which panel is which condition, dataset or method; that is usually given only
in the caption, a shared legend or the panel's position ("left: MNIST, right: CIFAR-100"). The readings come
back as unlabeled tables that the answering model has to match to the caption itself. A per-panel change in
MinerU would have the same problem, unless each panel were read with the caption, or the whole figure were
described once after its panels.

