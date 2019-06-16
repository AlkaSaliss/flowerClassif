from bokeh.layouts import gridplot
from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource


def plot_training_history(history):

    source = ColumnDataSource(data=dict(
        acc=history.history['acc'],
        val_acc=history.history['val_acc'],
        loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        epochs=range(1, len(history.history['acc']) + 1)
    ))

    TOOLTIPS = [
        ("epochs", "@epochs"),
        ("accuracy", "@acc"),
        ("val_accuracy", "@val_acc")
    ]

    f1 = figure(title='Training and validation accuracy',
                plot_width=500, plot_height=300)
    f1.line(
        x='epochs',
        y='acc',
        source=source,
        color="#ff9900",
        tooltips=TOOLTIPS,
        legend='Train'
    )
    f1.line(
        x='epochs',
        y='val_acc',
        source=source,
        color="#0000e6",
        tooltips=TOOLTIPS,
        legend='Validation'
    )

    TOOLTIPS = [
        ("epochs", "@epochs"),
        ("loss", "@loss"),
        ("val_loss", "@val_loss")
    ]
    f2 = figure(title='Training and validation loss',
                plot_width=500, plot_height=300)
    f2.line(
        x='epochs',
        y='loss',
        source=source,
        color="#ff9900",
        tooltips=TOOLTIPS,
        legend="Training"
    )
    f2.line(
        x='epochs',
        y='val_loss',
        source=source,
        color="#0000e6",
        tooltips=TOOLTIPS,
        legend="Validation"
        )

    p = gridplot([f1, f2])
    show(p)
