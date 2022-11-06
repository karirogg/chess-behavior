# chess-behaviour

# TODO

## Gagnavinnsla

- [-] Búa til E2E jipeline sem les inn gögn og skilar gögnum á ágætu formi
- [-] Velja notendur til að sækja frá
- [-] Nota `torch.DataLoader` til þess að geyma gögnin (padded) í batches

## Líkanagerð

- [-] Setja upp beinagrind að líkani (E2E) með einföldu líkani í torch lightning
  - [-] Carlsen/ekki Carlsen
  - [-] k-shot pipeline
- [-] Tengja líkanið við WandB
- [-] Fá líkanið til þess að læra
  - [-] ResNet í conv skrefi
  - [-] Tune-a LSTM líkanið
- [-] Baseline líkan - Arnar
  - [-] Ekki time dependency

# Spurningar

- Hversu mikil áhrif hefur tíma factor?
- Þurfum við að skoða alla skákina? Eða nægir að skoða opnun?
- Greina svindl?
