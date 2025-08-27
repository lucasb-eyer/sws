#!/usr/bin/fish

# I'm not sure which one is AGI, but I know the rules...
for d in 16 24 32
  python -m example.main --config example/super_agi.py \
    lr="1e-3 / ($d / 8)" \
    model.depth=$d \
    model.heads="$d // 2"
end
