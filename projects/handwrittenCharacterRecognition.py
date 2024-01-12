import logging
# Add this line before the PyTorch log messages appear
logging.getLogger('yolov5').setLevel(logging.CRITICAL)

def pil_to_b64(im, enc_format="png", **kwargs):
    from io import BytesIO
    import base64
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded

def handwritten_character_recognition(image_path):
    from shiftlab_ocr.doc2text.reader import Reader
    reader = Reader()
    result = reader.doc2text(image_path)
    rv = {}
    rv["recognized text"] = result[0]
    rv["handwritten word images"] = {}
    for k,images in enumerate(result[1]):
        rv["handwritten word images"][f"img_{k}"] = pil_to_b64(images.img)
    return rv

def create_app(b64_img_dict):
    from dash import Dash, html
    from dash import Dash, dcc, html, Input, Output,callback
    import dash_bootstrap_components as dbc 
    app = Dash(__name__)

    
    
    # components = [  
        # html.Div(
            # html.Img(id=f"{k}",className="image", src="data:image/png;base64, " + b64_img)
        # )
        # for k,b64_img in b64_img_dict.items()
    # ]
    dd = [k for k,b64_img in b64_img_dict.items()]
   
    app.layout = html.Div([
        dcc.Dropdown(dd, dd[0], id='demo-dropdown'),
        html.Div(id='dd-output-container')
    ])
    # print(app.layout)
    @callback(
        Output('dd-output-container', 'children'),
        Input('demo-dropdown', 'value')
    )
    def update_output(k):
        b64_img = b64_img_dict[k]
        print(b64_img)
        return html.Div(
            dbc.Col([
                dbc.Row(
                    html.Img(id=f"{k}",className="image", src="data:image/png;base64, " + b64_img)
                )
            ])
        )

    return app

if __name__=="__main__":
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/konverner/shiftlab_ocr/main/demo_image.png','test.png')
    image_path = "test.png"
    from pprint import pprint
    rv = handwritten_character_recognition(image_path)
    # pprint(rv)
    app = create_app(rv["handwritten word images"])
    app.run(debug = True)