
# aguvis json file with mobile action space
MOBILE_FILE = [
    "android_control.json",
    "gui-odyssey-l1.json",
    "aitw-l3.json",
    "coat.jsonamex-l2.json",
    "amex-l1.json",
    "amex-l3.json",
    "gui-odyssey-l3.json",
    "aitw-l1.json",
    "aitw-l2.json",
    "gui-odyssey-l2.json",
]

# Processing:  guienv
# Max conversations by image: 5 conversations
# Duplicates: 0
# Duplicate images in guienv.json. difference: 257578
# len(images_path): 327972
# len(images_set_path): 70394
# user/assistant by image: 3.6590902633747193
# 
# Processing:  omniact
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in omniact.json
# len(images_path): 6720
# len(images_set_path): 6720
# user/assistant by image: 0.0
# 
# Processing:  ricoig16k
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in ricoig16k.json
# len(images_path): 16133
# len(images_set_path): 16133
# user/assistant by image: 0.0
# 
# Processing:  ricosca
# Max conversations by image: 20 conversations
# Duplicates: 0
# Duplicate images in ricosca.json. difference: 155066
# len(images_path): 173212
# len(images_set_path): 18146
# user/assistant by image: 8.54546456519343
# 
# Processing:  seeclick
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in seeclick.json
# len(images_path): 271121
# len(images_set_path): 271121
# user/assistant by image: 0.0
# 
# Processing:  webui350k
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in webui350k.json
# len(images_path): 57389
# len(images_set_path): 57389
# user/assistant by image: 0.0
# 
# Processing:  ui_refexp
# Max conversations by image: 15 conversations
# Duplicates: 32
# Duplicate images in ui_refexp.json. difference: 10978
# len(images_path): 15624
# len(images_set_path): 4646
# user/assistant by image: 2.3628928110202323
# 
# Processing:  widget_captioning
# Max conversations by image: 161 conversations
# Duplicates: 4877
# Duplicate images in widget_captioning.json. difference: 87017
# len(images_path): 101426
# len(images_set_path): 14409
# user/assistant by image: 6.039072801721146
# 
# total_samples = 458958

config_dict_stage_1 = [
    {
        "json_path": "guienv.json",
        "images_folder": "guienvs/images/",
    },
    {
        "json_path": "omniact.json",
        "images_folder": "omniact/images/",
    },
    {
        "json_path": "ricoig16k.json",
        "images_folder": "ricoig16k/images/",
    },
    {
        "json_path": "ricosca.json",
        "images_folder": "ricosca/images/",
    },
    {
        "json_path": "seeclick.json",
        "images_folder": "seeclick/seeclick_web_imgs/",
    },
    {
        "json_path": "webui350k.json",
        "images_folder": "webui350k/images/",
    },
    {
        "json_path": "ui_refexp.json",
        "images_folder": "ui_refexp/images/",
    },
    {
        "json_path": "widget_captioning.json",
        "images_folder": "widget_captioning/images/",
    },
    
]


# Processing:  mind2web-l3
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in mind2web-l3.json
# len(images_path): 7591
# len(images_set_path): 7591
# user/assistant by image: 0.0
# 
# Processing:  guiact-web-single
# Max conversations by image: 12 conversations
# Duplicates: 0
# Duplicate images in guiact-web-single.json. difference: 54134
# len(images_path): 67396
# len(images_set_path): 13262
# user/assistant by image: 4.081888101342181
# 
# Processing:  guiact-web-multi-l3
# Max conversations by image: 2 conversations
# Duplicates: 0
# Duplicate images in guiact-web-multi-l3.json. difference: 24
# len(images_path): 16704
# len(images_set_path): 16680
# user/assistant by image: 0.0014388489208633094
# 
# Processing:  miniwob-l3
# Max conversations by image: 6 conversations
# Duplicates: 0
# Duplicate images in miniwob-l3.json. difference: 161
# len(images_path): 9826
# len(images_set_path): 9665
# user/assistant by image: 0.016658044490429385
# 
# Processing:  coat
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in coat.json
# len(images_path): 11921
# len(images_set_path): 11921
# user/assistant by image: 0.0
# 
# Processing:  android_control
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in android_control.json
# len(images_path): 74714
# len(images_set_path): 74714
# user/assistant by image: 0.0
# 
# Processing:  gui-odyssey-l3
# Max conversations by image: 2 conversations
# Duplicates: 0
# Duplicate images in gui-odyssey-l3.json. difference: 24
# len(images_path): 118282
# len(images_set_path): 118258
# user/assistant by image: 0.0002029461008980365
# 
# Processing:  amex-l3
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in amex-l3.json
# len(images_path): 38469
# len(images_set_path): 38469
# user/assistant by image: 0.0
# 
# Processing:  aitw-l3
# Max conversations by image: 0 conversations
# Duplicates: 0
# No duplicate images in aitw-l3.json
# len(images_path): 18992
# len(images_set_path): 18992
# user/assistant by image: 0.0
# 
# Total samples: 309552


config_dict_stage_2 = [
    {
        "json_path": "mind2web-l3.json",
        "images_folder": "mind2web/",
    },
    {
        "json_path": "guiact-web-single.json",
        "images_folder": "guiact-web-single/images/",
    },
    {
        "json_path": "guiact-web-multi-l3.json",
        "images_folder": "guiact-web-multi-v2/images",
    },
    {
        "json_path": "miniwob-l3.json",
        "images_folder": "images",
    },
    {
        "json_path": "coat.json",
        "images_folder": "coat/images/",
    },
    {
        "json_path": "android_control.json",
        "images_folder": "android_control/images/",
    },
    {
        "json_path": "gui-odyssey-l3.json",
        "images_folder": "gui-odyssey/images/",
    },
    {
        "json_path": "amex-l3.json",
        "images_folder": "amex/images/",
    },
    {
        "json_path": "aitw-l3.json",
        "images_folder": "aitw-v1/images/",
    },
]
