var NN = (function () {

	// vector arithmetic
	function dot(v, w) {
		// dot product, truncating to shorter vector
		var l = Math.min(v.length, w.length);
		var result = 0;
		for (var i = 0; i < l; ++i) {
			result += v[i] * w[i];
		}
		return result;
	}
	function vplus(v, w) {
		return v.map((x, i) => x + w[i]);
	}
	function vminus(v, w) {
		return v.map((x, i) => x - w[i]);
	}
	function vtimes(v, w) {
		return v.map((x, i) => x * w[i]);
	}
	function vscale(a, v) {
		return v.map((x) => a * x);
	}
	function tscale(a, t) {
		if (t instanceof Array) {
			return t.map((x) => tscale(a, x));
		} else {
			return a * t;
		}
	}
	function tadd(t1, t2) {
		if (t1 instanceof Array) {
			return t1.map((x, i) => tadd(x, t2[i]));
		} else {
			return t1 + t2;
		}
	}
	function otimes(v, w) {
		return v.map((vi) => vscale(vi, w));
	}
	function vcopy(v) {
		return v.map((x) => x);
	}
	function combo_rows(v, m) {
		// return w = vT m
		if (v.length < 1) { return []; }
		var w = vscale(v[0], m[0]);
		for (var i = 1; i < v.length; ++i) {
			m[i].forEach(function (mij, j) {
				w[j] += v[i] * mij;
			});
		}
		return w;
	}

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
		this.weight = [];
		for (var i = 0; i < dim_to; ++i) {
			var weight_row = [];
			for (var j = 0; j <= dim_from; ++j) {
				weight_row.push(randomize());
			}
			this.weight.push(weight_row);
		}
	}
	Layer.prototype.call = function (v) {
		return this.weight.map((wi) => wi[v.length] + dot(wi, v));
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
		for (var i = 0; i < layers.length; ++i) {
			layer_outputs.push(layers[i].call(layer_inputs[i]));
			layer_inputs.push(layer_outputs[i].map(layers[i].activation));
		}
		// derivative of L2 objective w.r.t. current position in the network
		var dee = vminus(layer_inputs[layers.length], y);
		for (var i = layers.length - 1; i >= 0; --i) {
			// absorb activation function derivative pointwise
			layer_outputs[i].forEach(function (o, j) {
				dee[j] *= layers[i].activation.d(o);
			});
			// output_i = w_ij [input_j, 1]
			//     d output_i = dw_ij [input_j, 1] + w_ij d input_j
			//     d objective / dw_ij = dee_i [input_j, 1]
			layer_inputs[i].push(1);  // capture bias term
			layer_gradients[i] = otimes(dee, layer_inputs[i]);
			// d objective / d input_j = dee_i w_ij (without last column)
			dee = combo_rows(dee, layers[i].weight);
			dee.pop();  // drop bias term
		}
		return layer_gradients;
	};
	Net.prototype.zerograd = function () {
		return this.layers.map(function (layer) {
			return layer.weight.map(function (row) {
				return row.map(function (item) {
					return 0;
				})
			})	
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
			var dim_from = this.layers[ell].weight[0].length - 1;
			var dim_to = this.layers[ell].weight.length;
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

	return {
		"Net": Net,
		"relu": relu,
		"tanh": tanh,
		"nop": nop,
		"leaky_relu": leaky_relu(0.2),
		"softplus": softplus,
		"abs": leaky_relu(-1),
		"square": square,
		"gaussian": gaussian,
	};
})();
