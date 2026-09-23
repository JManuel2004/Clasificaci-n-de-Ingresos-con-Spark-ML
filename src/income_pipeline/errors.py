"""Pipeline errors that callers can catch and report."""


class DataQualityError(Exception):
    """Raised when a dataset breaks the training or scoring contract."""

    def __init__(self, report: dict):
        self.report = report
        reasons = "; ".join(report.get("errors") or [])
        super().__init__(reasons or "data quality check failed")
