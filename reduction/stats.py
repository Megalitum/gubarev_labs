import timeit
import sklearn.metrics.regression as reg_metrics


def response_fit_report(reference, response, name):
    r2 = reg_metrics.r2_score(reference, response)
    print(r2)


def print_stats(method, T, err_gen, eps, delta, h_low, h_up, task, times):
    """
    Returns delta, cover_percent, match_percent, r2, stdev, time.  
    """
    err_name = err_gen.__name__
    h_real, series_real = generate_pure_h(T, 0.01)
    series = np.arange(0, delta * len(h_low), delta)
    assert(len(series) == len(h_low))
    match_percent, cover_percent, r2, stdev, *other = test_h_prediction(series_real, h_real, series, h_low, h_up)
    time = timeit.timeit('measured_task()', number=times, globals=dict(measured_task=task)) / times
    # print('Method: %d; T: %0.2f; Error: %s; Eps: %0.2f; Delta: %0.5f; Coverage: %0.2f%%; Match: %0.2f%%; R2: %0.4f; Std: %0.4f; Avg.time: %fs' %
    #      (method, T, err_name, eps, delta, cover_percent, match_percent, r2, stdev, time))
    return delta, cover_percent, match_percent, r2, stdev, time, other


def test_h_prediction(series_real, h_real, series, h_low, h_up):
    l, r = np.searchsorted(series_real, series[0]), np.searchsorted(series_real, series[-1], side='right')
    series_real = series_real[l:r]
    cover_percent = len(series_real) / len(h_real) * 100
    h_real = h_real[l:r]
    h_low, h_up = np.interp(series_real, series, h_low), np.interp(series_real, series, h_up)
    match_percent = calculate_match_percent(h_real, h_low, h_up)
    h_cent = (h_low + h_up) / 2
    r2 = r2_score(h_real, h_cent)
    stdev = np.sqrt(np.sum((h_up - h_low)**2) / (len(h_low) - 1))
    return match_percent, cover_percent, r2, stdev, series_real, np.abs(h_real- h_cent)


def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%0.5f' % height,
                ha='center', va='bottom')

def autolabel_inside(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%0.5f' % height,
                ha='center', va='top')


def method_analyzer(T, delta, error, ignore_second=False):
    # params : cover_percent, match_percent, r2, stdev, time, h_series, abs(h_real - h_pred)
    delta, *params1 = run_m1(T=T, eps=0.5, delta=delta, err_gen=error)
    if not ignore_second:
        delta, *params2 = run_m2(T=T, eps=0.5, delta=delta, err_gen=error)
    depth = int(np.floor(np.log2(T / delta))) - 1
    delta, *params3 = run_m3(T=T, eps=0.5, depth=depth, J=5, err_gen=error)
    params = [params1, params3] if ignore_second else [params1, params2, params3]

    # Show coverage and matching
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    index = np.arange(2)
    bar_width = 0.35 if ignore_second else 0.25
    opacity = 0.4
    rects1 = ax1.bar(index, params[0][:2], bar_width,
                     alpha=opacity,
                     color='b',
                     label='Метод 1')
    autolabel_inside(ax1, rects1)
    if not ignore_second:
        rects2 = ax1.bar(index + bar_width, params[1][:2], bar_width,
                         alpha=opacity,
                         color='r',
                         label='Метод 2')
        autolabel_inside(ax1, rects2)
        rects3 = ax1.bar(index + 2 * bar_width, params[-1][:2], bar_width,
                         alpha=opacity,
                         color='g',
                         label='Метод 3')
    else:
        rects3 = ax1.bar(index + bar_width, params[-1][:2], bar_width,
                         alpha=opacity,
                         color='g',
                         label='Метод 3')
    autolabel_inside(ax1, rects3)
    ax1.set_xlabel('Статистики')
    ax1.set_ylabel('%')
    ax1.set_title('Статистики покрытия и попадания в доверительные интервалы %s' % (error.__name__))
    if ignore_second:
        ax1.set_xticks(index + bar_width)
    else:
        ax1.set_xticks(index + 1.5 * bar_width)
    ax1.set_xticklabels(('Покрытие', 'Попадание'))
    ax1.legend(loc=0)
    ax1.grid(False)
    fig1.tight_layout()

    # Show r2, stdev, time
    fig2, (ax21, ax22, ax23) = plt.subplots(1, 3, figsize=(12, 5))
    index = np.arange(len(params))
    xlabels = ('Метод 1', 'Метод 2', 'Метод 3') if not ignore_second else ('Метод 1', 'Метод 3')
    bar_width = 0.5
    ax21.set_title('$R^2$')
    ax21.scatter(index, [p[2] for p in params], marker='^', s=40, color='r')
    ax21.set_xticks(index)
    ax21.set_xticklabels(xlabels)
    ax22.set_title('Std')
    ax22.grid(False)
    rects2 = ax22.bar(index, [p[3] for p in params], bar_width,
                      alpha=opacity,
                      color='b')
    autolabel(ax22, rects2)
    ax22.set_xticks(index + 0.5 * bar_width)
    ax22.set_xticklabels(xlabels)
    ax23.set_title('Время работы')
    ax23.grid(False)
    rects3 = ax23.bar(index, [p[4] for p in params], bar_width,
                      alpha=opacity,
                      color='b')
    autolabel(ax23, rects3)
    ax23.set_ylabel('s')
    ax23.set_xticks(index + 0.5 * bar_width)
    ax23.set_xticklabels(xlabels)
    fig2.tight_layout()
    # Show residual
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    if not ignore_second:
        h_pair1, h_pair2, h_pair3 = (p[-1] for p in params)
    else:
        h_pair1, h_pair3 = (p[-1] for p in params)
    ax3.set_title('Невязка: abs(h_real - h_central)')
    ax3.plot(h_pair1[0], h_pair1[1], 'r', label='$|\Delta h|_1$')
    if not ignore_second:
        ax3.plot(h_pair2[0], h_pair2[1], 'g', label='$|\Delta h|_2$')
    ax3.plot(h_pair3[0], h_pair3[1], 'b', label='$|\Delta h|_3$')
    ax3.legend(loc=0)
    fig3.tight_layout()