from flair.visual.training_curves import Plotter

clf_dir = 'resources/binary_unbiased_031219/'
plotter = Plotter()
plotter.plot_training_curves('./resources/loss.tsv')
plotter.plot_weights(clf_dir + 'weights.txt')
