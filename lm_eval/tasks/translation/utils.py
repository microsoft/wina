import argparse

import yaml


try:
    import pycountry
except ModuleNotFoundError:
    raise Exception(
        "`pycountry` is required for generating translation task prompt templates. \
please install pycountry via pip install lm-eval[multilingual] or pip install -e .[multilingual]",
    )


# Different translation benchmarks included in the library. Mostly WMT.
# These correspond to dataset names (subsets) on HuggingFace for each dataset.
# A yaml file is generated by this script for each language pair.

gpt3_translation_benchmarks = {
    "wmt14": ["fr-en"],  # ["en-fr", "fr-en"],  # French
    "wmt16": [
        "ro-en",
        "de-en",
    ],  # ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}

# 28 total
LANGUAGES = {
    **gpt3_translation_benchmarks,
    # "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt2017": ["en-ar"],  # Arabic
}


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name


def gen_lang_yamls(output_dir: str, overwrite: bool) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in LANGUAGES.keys():
        for dataset_name in LANGUAGES[lang]:
            src_lang, _, tgt_lang = dataset_name.partition("-")
            for src, tgt in [[src_lang, tgt_lang], [tgt_lang, src_lang]]:
                # both translation directions for each lang pair
                lang_pair = src + "-" + tgt
                file_name = f"{lang}_{lang_pair}.yaml"
                try:
                    source, target = code_to_language(src), code_to_language(tgt)

                    groups = ["generate_until", "translation", lang]
                    if lang in gpt3_translation_benchmarks.keys():
                        groups += ["gpt3_translation_benchmarks"]

                    with open(
                        f"{output_dir}/{file_name}",
                        "w" if overwrite else "x",
                        encoding="utf8",
                    ) as f:
                        f.write("# Generated by utils.py\n")
                        yaml.dump(
                            {
                                "include": "wmt_common_yaml",
                                "group": groups,
                                "dataset_path": lang,
                                "dataset_name": dataset_name
                                if not (lang == "iwslt2017")
                                else "iwslt2017-" + dataset_name,
                                "task": f"{lang}-{lang_pair}",
                                "doc_to_text": f"{source} phrase: "
                                + "{{translation["
                                + f'"{src}"'
                                + "]}}\n"
                                + f"{target} phrase:",
                                "doc_to_target": " {{"
                                + "translation["
                                + f'"{tgt}"]'
                                + "}}",
                            },
                            f,
                        )
                except FileExistsError:
                    err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write yaml files to"
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
