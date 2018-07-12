import math
import torch
import warnings
import itertools
import numbers

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence


class RNNBase(Module):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            weight_arr = list(itertools.chain.from_iterable(self.all_weights))
            weight_stride0 = len(self.all_weights[0])

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on weight_arr, that's why the
                # no_grad() is necessary.
                weight_buf = torch._cudnn_rnn_flatten_weight(
                    weight_arr, weight_stride0,
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

            self._param_buf_size = weight_buf.size(0)
            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(RNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        print('1111111111111111111111111111111111111111111')
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            
        for x in self._all_weights:
            print(_all_weights) 
       
            
        if not hasattr(self._all_weights,'org'):
            self._all_weights.org=self._all_weights.data.clone()
        self._all_weights.data=binarize(self.all_weights.org)
        
       

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        self.check_forward_args(input, hx, batch_sizes)
        func = self._backend.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            dropout_state=self.dropout_state,
            variable_length=is_packed,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class RNN(RNNBase):


    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


class LSTM(RNNBase):


    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class BGRU(RNNBase):
   

    def __init__(self, *args, **kwargs):
        super(BGRU, self).__init__('GRU', *args, **kwargs)


class RNNCellBase(Module):

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class RNNCell(RNNCellBase):
    

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        #self.weight_ih=binarize(w_ih)
        #self.weight_hh=(w_hh)
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        if self.nonlinearity == "tanh":
            func = self._backend.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = self._backend.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        if not hasattr(self.weight_ih,'org'):
            self.weight_ih.org=self.weight_ih.data.clone()
        self.weight_ih.data=binarize(self.weight_ih.org)
      

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class LSTMCell(RNNCellBase):
   
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self._backend.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class GRUCell(RNNCellBase):
   

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        return self._backend.GRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
