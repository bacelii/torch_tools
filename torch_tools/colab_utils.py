
def set_output_cell_size(
    maxHeight=300):
    """
    To shrink the size of the output cell

    """
    from google.colab.output import eval_js
    eval_js(f'google.colab.output.setIframeHeight("{maxHeight}")')  