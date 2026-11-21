import unittest
from unittest.mock import Mock

from raw2film.utils import add_metadata


class AddMetadataTests(unittest.TestCase):
    def test_does_not_copy_source_compression_tag(self):
        exiftool = Mock()

        add_metadata(
            exiftool,
            "output.tif",
            {
                "EXIF:Compression": 7,
                "EXIF:Make": "Camera",
            },
            exp_comp=0,
        )

        tags = exiftool.set_tags.call_args.args[1]
        self.assertNotIn("EXIF:Compression", tags)
        self.assertEqual(tags["EXIF:Make"], "Camera")


if __name__ == "__main__":
    unittest.main()
