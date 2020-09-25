let { Matrix, determinant } = require('ml-matrix');

module.exports = function conhull(x, y) {
    x = Matrix.checkMatrix(x);
    y = Matrix.checkMatrix(y);
    if (x.rows === 1) x = x.transpose();
    if (y.rows === 1) y = y.transpose();
    let m = x.rows - 1;
    if (x.rows !== y.rows) {
        return
    }
    if (m === 1) return [0, 1];

    if (m === 0) return [0];

    let start = 0;
    let v = 0;
    let w = x.rows - 1;
    let h = new Matrix(m + 1, 1);
    for (let i = 0; i < h.rows; i++) {
        h.set(i, 0, i);
    }
    // console.log(h.rows, y.rows, x.rows)
    let flag = 0;
    while (next(v, m) !== start || flag === 0) {
        if (next(v, m) === w) flag = 1;

        let a = v;
        let b = next(v, m);
        let c = next(next(v, w), m);
    
        let matrix = new Matrix([ [ x.get(a, 0), y.get(a, 0), 1 ], [ x.get(b, 0), y.get(b, 0), 1 ], [x.get(c, 0), y.get(c, 0), 1 ] ]);

        if (determinant(matrix) >= 0) {
            v = next(v, m);
        } else {
            let j = next(v, m);
            x.removeRow(j);
            y.removeRow(j);
            h.removeRow(j);
            m -= 1;
            w -= 1;
            v = pred(v, m);
        }
    }

    return h.to1DArray();
}

function next(v, m) {
    return v === m ? 0 : v + 1;
}

function pred(v, m) {
    return v === 0 ? m : v - 1;
}
