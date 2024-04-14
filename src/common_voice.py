""" Common Voice Dataset"""

import csv
import json
import os

import datasets
from datasets.utils.py_utils import size_str
from tqdm import tqdm

# from .languages import LANGUAGES

# from .release_stats import STATS

STATS = {
    "bundleURLTemplate": "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-13.0-2023-03-09/{locale}.tar.gz",
    "locales": {
        "fa": {
            "buckets": {
                "dev": 10440,
                "invalidated": 14071,
                "other": 20673,
                "reported": 2268,
                "test": 10440,
                "train": 28024,
                "validated": 320143,
            },
            "reportedSentences": 2259,
            "duration": 1415098376,
            "clips": 354887,
            "splits": {
                "accent": {"": 1},
                "age": {
                    "": 0.25,
                    "twenties": 0.31,
                    "thirties": 0.37,
                    "fifties": 0.02,
                    "fourties": 0.02,
                    "teens": 0.03,
                    "sixties": 0,
                },
                "gender": {"": 0.22, "male": 0.71, "female": 0.07, "other": 0},
            },
            "users": 4188,
            "size": 10368977174,
            "checksum": "921ff70850b58468bcc232f1d6f8e7c5bf58aff2ee1efdd4f26de19e75f7ed2a",
            "avgDurationSecs": 3.987,
            "validDurationSecs": 1276558.001,
            "totalHrs": 393.08,
            "validHrs": 354.59,
        },
    },
    "totalDuration": 97709611853,
    "totalValidDurationSecs": 63681475,
    "totalHrs": 27141,
    "totalValidHrs": 17689,
    "version": "13.0.0",
    "date": "2022-03-15",
    "name": "Common Voice Corpus 13",
    "multilingual": True,
}

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"

# TODO: change "streaming" to "main" after merge!
_BASE_URL = (
    "https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/resolve/main/"
)

_AUDIO_URL = _BASE_URL + "audio/{lang}/{split}/{lang}_{split}_{shard_idx}.tar"

_TRANSCRIPT_URL = _BASE_URL + "transcript/{lang}/{split}.tsv"

_N_SHARDS_URL = _BASE_URL + "n_shards.json"


class CommonVoiceConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, name, version, **kwargs):
        self.language = kwargs.pop("language", None)
        self.release_date = kwargs.pop("release_date", None)
        self.num_clips = kwargs.pop("num_clips", None)
        self.num_speakers = kwargs.pop("num_speakers", None)
        self.validated_hr = kwargs.pop("validated_hr", None)
        self.total_hr = kwargs.pop("total_hr", None)
        self.size_bytes = kwargs.pop("size_bytes", None)
        self.size_human = size_str(self.size_bytes)
        description = (
            f"Common Voice speech to text dataset in {self.language} released on {self.release_date}. "
            f"The dataset comprises {self.validated_hr} hours of validated transcribed speech data "
            f"out of {self.total_hr} hours in total from {self.num_speakers} speakers. "
            f"The dataset contains {self.num_clips} audio clips and has a size of {self.size_human}."
        )
        super(CommonVoiceConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            description=description,
            **kwargs,
        )


class CommonVoice(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 1000

    BUILDER_CONFIGS = [
        CommonVoiceConfig(
            name="fa",  # lang,
            version=STATS["version"],
            language="Persian",  # LANGUAGES[lang],
            release_date=STATS["date"],
            num_clips=lang_stats["clips"],
            num_speakers=lang_stats["users"],
            validated_hr=float(lang_stats["validHrs"])
            if lang_stats["validHrs"]
            else None,
            total_hr=float(lang_stats["totalHrs"]) if lang_stats["totalHrs"] else None,
            size_bytes=int(lang_stats["size"]) if lang_stats["size"] else None,
        )
        for lang, lang_stats in STATS["locales"].items()
    ]

    def _info(self):
        total_languages = len(STATS["locales"])
        total_valid_hours = STATS["totalValidHrs"]
        description = (
            "Common Voice is Mozilla's initiative to help teach machines how real people speak. "
            f"The dataset currently consists of {total_valid_hours} validated hours of speech "
            f" in {total_languages} languages, but more voices and languages are always added."
        )
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=48_000),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
                "variant": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=description,
            features=features,
            supervised_keys=None,
            version=self.config.version,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name
        n_shards_path = dl_manager.download_and_extract(_N_SHARDS_URL)
        with open(n_shards_path, encoding="utf-8") as f:
            n_shards = json.load(f)

        audio_urls = {}
        splits = ("train", "dev", "test", "other", "invalidated")
        for split in splits:
            audio_urls[split] = [
                _AUDIO_URL.format(lang=lang, split=split, shard_idx=i)
                for i in range(n_shards[lang][split])
            ]
        archive_paths = dl_manager.download(audio_urls)
        local_extracted_archive_paths = (
            dl_manager.extract(archive_paths) if not dl_manager.is_streaming else {}
        )

        meta_urls = {
            split: _TRANSCRIPT_URL.format(lang=lang, split=split) for split in splits
        }
        meta_paths = dl_manager.download_and_extract(meta_urls)

        split_generators = []
        split_names = {
            "train": datasets.Split.TRAIN,
            "dev": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        for split in splits:
            split_generators.append(
                datasets.SplitGenerator(
                    name=split_names.get(split, split),
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get(
                            split
                        ),
                        "archives": [
                            dl_manager.iter_archive(path)
                            for path in archive_paths.get(split)
                        ],
                        "meta_path": meta_paths[split],
                    },
                ),
            )

        return split_generators

    def _generate_examples(self, local_extracted_archive_paths, archives, meta_path):
        data_fields = list(self._info().features.keys())
        metadata = {}
        with open(meta_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in tqdm(reader, desc="Reading metadata..."):
                if not row["path"].endswith(".mp3"):
                    row["path"] += ".mp3"
                # accent -> accents in CV 8.0
                if "accents" in row:
                    row["accent"] = row["accents"]
                    del row["accents"]
                # if data is incomplete, fill with empty values
                for field in data_fields:
                    if field not in row:
                        row[field] = ""
                metadata[row["path"]] = row

        for i, audio_archive in enumerate(archives):
            for path, file in audio_archive:
                _, filename = os.path.split(path)
                if filename in metadata:
                    result = dict(metadata[filename])
                    # set the audio feature and the path to the extracted file
                    path = (
                        os.path.join(local_extracted_archive_paths[i], path)
                        if local_extracted_archive_paths
                        else path
                    )
                    result["audio"] = {"path": path, "bytes": file.read()}
                    result["path"] = path
                    yield path, result
