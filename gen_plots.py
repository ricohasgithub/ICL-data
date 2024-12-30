from util import gen_qk_weights_gif

run_name = "Disentangled|Readout|K=512|L=32|p_B=0.375|p_C=0.375|B=1|eps=0|00d1c8e2-9513-482c-ac3f-fb8260b8eb13"

gen_qk_weights_gif(
    run_name,
    1,
)
gen_qk_weights_gif(
    run_name,
    2,
)
gen_qk_weights_gif(
    run_name,
    "out",
)
