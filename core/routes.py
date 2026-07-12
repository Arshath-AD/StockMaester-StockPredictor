# ---------------------------------------------------------------------------
# core/routes.py — Flask Blueprint: all HTTP routes
# ---------------------------------------------------------------------------

from flask import Blueprint, render_template, request

from .config import STOCK_MAP
from .predictor import get_prediction

main = Blueprint("main", __name__)


@main.route("/", methods=["GET", "POST"])
def index():
    prediction     = None
    accuracy       = None
    selected_stock = None
    error_message  = None

    if request.method == "POST":
        stock_name = request.form.get("stock_name")

        if stock_name in STOCK_MAP:
            symbol         = STOCK_MAP[stock_name]
            selected_stock = stock_name

            result_signal, result_value = get_prediction(symbol)

            if result_signal is not None:
                prediction = result_signal
                accuracy   = f"{result_value * 100:.2f}%"
            else:
                error_message = result_value  # error string from predictor

    return render_template(
        "index.html",
        stocks         = STOCK_MAP.keys(),
        prediction     = prediction,
        accuracy       = accuracy,
        selected_stock = selected_stock,
        error          = error_message,
    )
