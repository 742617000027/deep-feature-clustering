class hparams:
    class dsp:
        sample_rate = 44100
        min_vol = -96
        max_vol = -0.3
        n_fft = 512
        hop_length = 128
        num_mels = 128
        num_audio_features = 10
        num_transient_spectrum_bins = 4 * int(0.01 * sample_rate)

    class training:
        batch_size = 1
        num_workers = 8
        accumulation_steps = 50
        validation_every_n_steps = 1000
        n_validation_steps = 200
        n_inference_examples = 5

    class model:
        class encoder:
            input_channels = 3
            expand_channels = 128
            lstm_hidden_dims = 2048
            lstm_num_layers = 3
            encoding_dims = 1024

        class decoder:
            lstm_hidden_dims = 512
            lstm_num_layers = 1

    files = 'path/to/files'
