import unittest

import csv_log


class CsvLogTest(unittest.TestCase):
    def test_generate_header_for_empty_config(self):
        config = []
        self.assertEqual(
            'TIME\n',
            csv_log.generate_header(config))

    def test_generate_header_for_single_mat_config(self):
        config = [{'width': 2, 'height': 3}]
        self.assertEqual(
            'TIME,S_0_0_0,S_0_0_1,S_0_0_2,S_0_1_0,S_0_1_1,S_0_1_2,S_0_mean\n',
            csv_log.generate_header(config))

    def test_generate_header_for_dual_mat_config(self):
        config = [{'width': 2, 'height': 3}, {'width': 2, 'height': 2}]
        self.assertEqual(
            'TIME,S_0_0_0,S_0_0_1,S_0_0_2,S_0_1_0,S_0_1_1,S_0_1_2,S_0_mean,S_1_0_0,S_1_0_1,S_1_1_0,S_1_1_1,S_1_mean\n',
            csv_log.generate_header(config))

    def test_generate_sensemat_configuration_comment(self):
        config = [{'width': 2, 'height': 3}]
        self.assertEqual(
            '# config = [{"width": 2, "height": 3}]\n',
            csv_log.generate_configuration_comment(config))

    def test_parse_sensemat_configuration_comment(self):
        csv_line = '# config = [{"width": 2, "height": 3}]\n'
        self.assertEqual(
            [{'width': 2, 'height': 3}],
            csv_log.parse_configuration_comment(csv_line))

    def test_parse_sensemat_configuration_comment_when_no_comment_line(self):
        csv_line = 'TIME,S_0_0_0,S_0_0_1,S_0_0_2,S_0_1_0,S_0_1_1,S_0_1_2,S_0_mean\n'
        self.assertEqual(
            None,
            csv_log.parse_configuration_comment(csv_line))


if __name__ == '__main__':
    unittest.main()
