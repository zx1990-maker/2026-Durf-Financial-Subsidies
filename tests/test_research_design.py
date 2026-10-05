import unittest

from run_counterfactual import classify_column, extract_header_info
from src.research_design import (
    classify_common_support,
    financial_subsidy_amount,
    financial_subsidy_rate,
    validate_out_of_time_window,
)


class ResearchDesignTests(unittest.TestCase):
    def test_subsidy_rate_and_amount_follow_paper_equations(self):
        rate = financial_subsidy_rate(0.071, 0.056)
        self.assertAlmostEqual(rate, 0.015)
        self.assertAlmostEqual(financial_subsidy_amount(rate, 1_000), 15.0)

    def test_negative_assets_are_rejected(self):
        with self.assertRaises(ValueError):
            financial_subsidy_amount(0.01, -1)

    def test_training_window_is_strictly_out_of_time(self):
        self.assertTrue(validate_out_of_time_window([2017, 2018, 2019], 2020))
        self.assertFalse(validate_out_of_time_window([2018, 2019, 2020], 2020))

    def test_support_threshold_boundaries(self):
        self.assertEqual(classify_common_support(1.20, 1.20, 1.40), "Good")
        self.assertEqual(classify_common_support(1.30, 1.20, 1.40), "Boundary")
        self.assertEqual(classify_common_support(1.41, 1.20, 1.40), "Extrapolation")

    def test_wind_header_parser(self):
        header = "资产总计\n[报告期] 2024年报\n[单位] 元"
        self.assertEqual(extract_header_info(header), ("资产总计", 2024))
        self.assertEqual(classify_column(header), ("total_assets", 2024))


if __name__ == "__main__":
    unittest.main()
