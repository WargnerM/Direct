function getErrorFunction(func, data, constants) {
    return (params) => {
      let y = func(data.x, params, constants);
      let errorValue = 0;
      for (let i = 0; i < data.x.length; i++) {
        errorValue += y[i] - data.y[i];
      }
      return Math.pow(errorValue, 2);
    };
  }

  module.exports = getErrorFunction;