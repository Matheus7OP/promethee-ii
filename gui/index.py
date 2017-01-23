from flask import render_template
from flask import request

from promethee.promethee_ii import PrometheeII
from flask import Flask

app = Flask('index')


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/run", methods=['POST'])
def run():
    alternatives = request.form.get('alternatives', '50')
    seed = request.form.get('seed', '123')
    criteria = [int(value) for key, value in request.form.items() if key.startswith('crit')]
    solver = PrometheeII(sample_size=int(alternatives), seed=int(seed), weights=criteria)
    results = solver.run()
    return render_template('results.html', data=results)


if __name__ == '__main__':
    app.run()
