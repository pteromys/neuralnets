var NN = (function () {

	function prod(numbers) {
		var answer = 1;
		for (var i = 0; i < numbers.length; ++i) {
			answer *= numbers[i];
		}
		return answer;
	}

	var ArrayType = Float64Array;
	function Matrix(rows, cols, existing_storage) {
		this.rows = rows;
		this.cols = cols;
		if (typeof existing_storage === 'undefined') {
			this.storage = new ArrayType(rows * cols);
		} else if (existing_storage instanceof ArrayType) {
			this.storage = new ArrayType(existing_storage);
		} else {
			this.storage = existing_storage;
		}
	}
	Matrix.from_vec = function from_vec(v) {
		return new Matrix(1, v.length, v);
	};
	Matrix.prototype = {
		// wrappers around TypedArray methods
		slice: function slice() {
			return new Matrix(this.rows, this.cols, this.storage.slice());
		},
		map: function map(f) {
			return new Matrix(this.rows, this.cols, this.storage.map(f));
		},
		imap: function imap(f) {
			for (var i = 0; i < this.storage.length; ++i) {
				this.storage[i] = f(this.storage[i], i, this.storage);
			}
		},
		// matrix entry access
		index: function index(row, col) {
			return row * this.cols + col;
		},
		read: function read(row, col) {
			return this.storage[row * this.cols + col];
		},
		write: function write(row, col, value) {
			return this.storage[row * this.cols + col] = value;
		},
		// matrix multiplication
		contractlast: function contractlast(right, out) {
			if (typeof out === 'undefined') {
				out = new Matrix(this.rows, right.rows);
			}

			// footgun 1: assume matrices are zero-padded out to infinity.
			// This gives up dimension-checking
			// but enables using extra cols to store whatever we like.
			var dimension = Math.min(this.cols, right.cols);

			for (var row = 0; row < out.rows; ++row) {
				var row_start = row * this.cols;
				for (var col = 0; col < out.cols; ++col) {
					// footgun 2: Add to outparam because that's usually what
					// we're doing. If you want to clobber, zero it yourself.
					var col_start = col * right.cols;
					var cell = out.read(row, col);
					for (var i = 0; i < dimension; ++i) {
						cell += this.storage[row_start + i] * right.storage[col_start + i];
					}
					out.write(row, col, cell);
				}
			}
			return out;
		},
		matmul: function matmul(right, out) {
			if (typeof out === 'undefined') {
				out = new Matrix(this.rows, right.cols);
			}

			// footgun 1: assume matrices are zero-padded out to infinity.
			// This gives up dimension-checking
			// but enables using extra cols to store whatever we like.
			var dimension = Math.min(this.cols, right.rows);

			for (var row = 0; row < out.rows; ++row) {
				var row_start = row * this.cols;
				for (var col = 0; col < out.cols; ++col) {
					// footgun 2: Add to outparam because that's usually what
					// we're doing. If you want to clobber, zero it yourself.
					var cell = out.read(row, col);
					for (var i = 0; i < dimension; ++i) {
						cell += this.storage[row_start + i] * right.storage[i * right.cols + col];
					}
					out.write(row, col, cell);
				}
			}
			return out;
		},
		// pointwise arithmetic
		pointwise_sub: function pointwise_sub(right, out) {
			if (typeof out === 'undefined') {
				out = new Matrix(this.rows, this.cols);
			}
			for (var i = 0; i < out.storage.length; ++i) {
				out.storage[i] = this.storage[i] - right.storage[i];
			}
			return out;
		},
	};

	function sample_normal() {
		var r = Math.sqrt(Math.abs(Math.log(1 - Math.random())));
		return r * Math.cos(Math.PI * Math.random());
	}
	function Layer(dim_from, dim_to, activation, init_scale) {
		// a neural network layer without activation
		// i.e. an affine function
		// bias is the last column of weight
		var randomize = () => init_scale * sample_normal() / Math.sqrt(dim_from);
		this.activation = activation;
		this.weight = new Matrix(dim_to, dim_from + 1);
		for (var row = 0; row < dim_to; ++row) {
			for (var col = 0; col <= dim_from; ++col) {
				this.weight.write(row, col, randomize());
			}
		}
	}
	Layer.prototype.call = function (v, out) {
		out = v.contractlast(this.weight, out);
		// add bias terms
		for (var i_out = 0; i_out < out.cols; ++i_out) {
			var bias = this.weight.read(i_out, v.cols);
			var row_end = (i_out + 1) * out.cols;
			for (var i_sample = 0; i_sample < out.rows; ++i_sample) {
				out.storage[i_sample * out.cols + i_out] += bias;
			}
		}
		return out;
	};

	function Net(dim_from, layer_sizes, dim_to, activation, init_scale) {
		this.layers = [];

		var last_size = dim_from;
		var layer_scale = Math.pow(init_scale, 1 / (layer_sizes.length + 1));
		for (var i = 0; i < layer_sizes.length; ++i) {
			this.layers.push(new Layer(last_size, layer_sizes[i], activation, layer_scale));
			last_size = layer_sizes[i];
		}
		this.layers.push(new Layer(last_size, dim_to, nop, layer_scale));
	}
	Net.prototype.call = function (v) {
		for (var i = 0; i < this.layers.length; ++i) {
			v = this.layers[i].call(v);
			v = v.map(this.layers[i].activation);
		}
		return v;
	};
	Net.prototype.l2grad = function (x, y) {
		var layers = this.layers;
		var layer_gradients = new Array(layers.length);  // T(objective) <- T(bias, weight)
		var layer_inputs = [x.slice()];
		var layer_outputs = [];
		var num_samples = x.rows;
		for (var i = 0; i < layers.length; ++i) {
			layer_outputs.push(layers[i].call(layer_inputs[i]));
			layer_inputs.push(layer_outputs[i].map(layers[i].activation));
		}
		// derivative of L2 objective w.r.t. current position in the network
		var output_gradient = layer_inputs[layers.length].pointwise_sub(y);
		for (var i = layers.length - 1; i >= 0; --i) {
			// absorb activation function derivative pointwise
			output_gradient.imap((d, j) => d * layers[i].activation.d(layer_outputs[i].storage[j]));
			// output_i = w_ij [input_j, 1]
			//     d output_i = dw_ij [input_j, 1] + w_ij d input_j
			//     d objective / dw_ij = output_gradient_i [input_j, 1]
			var grad = layer_gradients[i] = new Matrix(output_gradient.cols, layer_inputs[i].cols + 1);
			for (var row = 0; row < grad.rows; ++row) {
				for (var col = 0; col < layer_inputs[i].cols; ++col) {
					var g = 0;
					for (var sample = 0; sample < num_samples; ++sample) {
						g += output_gradient.read(sample, row) * layer_inputs[i].read(sample, col);
					}
					grad.write(row, col, g);
				}
				var g = 0;
				for (var sample = 0; sample < num_samples; ++sample) {
					g += output_gradient.read(sample, row);  // bias term
				}
				grad.write(row, layer_inputs[i].cols, g);
			}
			// d objective / d input_j = output_gradient_i w_ij (without last column)
			var new_output_gradient = new Matrix(num_samples, layers[i].weight.cols - 1);
			for (var sample = 0; sample < num_samples; ++sample) {
				for (var col = 0; col < new_output_gradient.cols; ++col) {
					var g = 0;
					for (var j = 0; j < output_gradient.cols; ++j) {
						g += output_gradient.read(sample, j) * layers[i].weight.read(j, col);
					}
					new_output_gradient.write(sample, col, g);
				}
			}
			output_gradient = new_output_gradient;
		}
		return layer_gradients;
	};
	Net.prototype.zerograd = function () {
		return this.layers.map(function (layer) {
			return layer.weight.map(function (w) {
				return 0;
			});
		});
	};
	Net.prototype.as_2d_shader = function () {
		lines = [
			"varying mediump vec2 vCoord;",
			"uniform sampler2D uWeights;",
			"mediump float unpack(mediump vec4 color) {",
			"    color = color * 255.;",
			"    mediump float sign = 1. - 2. * step(127.5, color.a);",
			"    mediump float exponent = 2. * mod(color.a, 128.) + step(127.5, color.b);",
			"    mediump float mantissa = ((color.r / 256. + color.g) / 256. + mod(color.b, 128.)) / 128. + 1.;",
			"    return sign * mantissa * exp2(exponent - 127.);",
			"}",
		];
		for (var ell = 0; ell < this.layers.length; ++ell) {
			lines.push(`mediump float activation${ell}${this.layers[ell].activation.glsl}`);
		}
		lines.push(
			"void main(void) {",
			"    mediump float v0[2]; v0[0] = vCoord.x; v0[1] = vCoord.y;",
			"    mediump vec2 weight_coord = vec2(1./1024., 1./1024.);",
		);
		for (var ell = 0; ell < this.layers.length; ++ell) {
			var dim_from = this.layers[ell].weight.cols - 1;
			var dim_to = this.layers[ell].weight.rows;
			lines.push(
				`mediump float v${ell+1}[${dim_to}];`,
				`for (int i = 0; i < ${dim_to}; ++i) {`,
				`    v${ell+1}[i] = 0.;`,
				`    for (int j = 0; j < ${dim_from}; ++j) {`,
				`        v${ell+1}[i] += v${ell}[j] * unpack(texture2D(uWeights, weight_coord));`,
				//"        gl_FragColor = vec4(gl_FragColor.gb, 0.5 * unpack(texture2D(uWeights, weight_coord)) + 0.5, 1.);",
				"        weight_coord.x += 1./512.;",
				"        weight_coord.y += floor(weight_coord.x) / 512.;",
				"        weight_coord.x = fract(weight_coord.x);",
				"    }",
				`    v${ell+1}[i] += unpack(texture2D(uWeights, weight_coord));`,
				`    v${ell+1}[i] = activation${ell}(v${ell+1}[i]);`,
				//"    gl_FragColor = vec4(gl_FragColor.gb, 0.5 * unpack(texture2D(uWeights, weight_coord)) + 0.5, 1.);",
				//"    if (vCoord.x < -0.5) { gl_FragColor = vec4(0.5 + 4096. * (weight_coord - vec2(259.5/512., 2.5/512.)), 0., 1.); }",
				"    weight_coord.x += 1./512.;",
				"    weight_coord.y += floor(weight_coord.x) / 512.;",
				"    weight_coord.x = fract(weight_coord.x);",
				"}",
			);
		}
		var last = this.layers.length;
		lines.push(
			`    mediump vec4 rgba = 0.5 * vec4(v${last}[0], v${last}[1], v${last}[2], v${last}[3]) + 0.5;`,
			"    gl_FragColor = vec4(rgba.rgb, 1.) * rgba.a;",
			//"gl_FragColor = vec4(0.5 - 1./1024., 1./255.5, 127./255. + 255./512., 1.);",
			//"    if (vCoord.x > 0.) { gl_FragColor = 0.5 * vec4(unpack(texture2D(uWeights, vec2(257.5/512., 2.5/512.))), unpack(texture2D(uWeights, vec2(258.5/512., 2.5/512.))), unpack(texture2D(uWeights, vec2(259.5/512., 2.5/512.))), 1.) + 0.5; }",
			"}",
		);
		return lines.join("\n");
	};

	function relu(x) { return Math.max(x, 0); }
	relu.d = function (x) { return Number(x >= 0); };
	relu.glsl = "(mediump float x) { return max(x, 0.); }";

	function leaky_relu(slope) {
		if (!(slope < 1)) { return nop; }
		var f = function (x) { return Math.max(x, slope * x); }
		f.d = function (x) { return (x >= 0) + (x < 0) * slope; };
		var slope_string = slope.toString();
		if (slope_string.indexOf('.') < 0) { slope_string += '.'; }
		f.glsl = `(mediump float x) { return max(x, ${slope_string} * x); }`;
		return f;
	}

	function tanh(x) {
		var e = Math.exp(x);
		var ei = 1/e;
		if (e == Infinity) { return 1; }
		if (ei == Infinity) { return -1; }
		return (e - ei) / (e + ei);
	}
	tanh.d = function (x) {
		var e = Math.exp(x);
		var sech = 2 / (e + 1/e);
		return sech * sech;
	};
	tanh.glsl = [
		"(mediump float x) {",
		"    mediump float ei2 = exp(-2. * abs(x));",
		"    return sign(x) * (1. - ei2) / (1. + ei2);",
		"}",
	].join("\n");

	function nop(x) { return x; }
	nop.d = function (x) { return 1; };
	nop.glsl = "(mediump float x) { return x; }";

	function softplus(x) { return Math.log(1 + Math.exp(x)); }
	softplus.d = function (x) { return 1 / (1 + Math.exp(-x)); };
	softplus.glsl = "(mediump float x) { return x < 9.704 ? log(1. + exp(x)) : x; }";

	function square(x) { return x * x; }
	square.d = function (x) { return 2 * x; }
	square.glsl = "(mediump float x) { return x * x; }";

	function gaussian(x) { return Math.exp(-0.5 * x * x); }
	gaussian.d = function (x) { return -x * gaussian(x); }
	gaussian.glsl = "(mediump float x) { return exp(-0.5 * x * x); }";

	function agnesi(x) { return 1 / (1 + x * x); }
	agnesi.d = function (x) { var a = agnesi(x); return -2 * x * a * a; }
	agnesi.glsl = "(mediump float x) { return 1. / (1. + x * x); }";

	return {
		"Matrix": Matrix,
		"Net": Net,
		"relu": relu,
		"tanh": tanh,
		"nop": nop,
		"leaky_relu": leaky_relu(0.2),
		"softplus": softplus,
		"abs": leaky_relu(-1),
		"square": square,
		"gaussian": gaussian,
		"agnesi": agnesi,
	};
})();
