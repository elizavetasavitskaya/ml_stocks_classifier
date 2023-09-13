from aiogram import Bot
from aiogram import executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.filters import Command
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import ParseMode
from matplotlib import pyplot as plt
from prettytable import prettytable

from update import load

TOKEN = "6557456309:AAEJNjcpux0WTtSMoyr4OReJ6da-PDzP6Rc"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
global action

indicators_buttons = [
    [types.KeyboardButton(text="RSI"), types.KeyboardButton(text="Stochastic RSI"),
     types.KeyboardButton(text="Bollinger Band")],
    [types.KeyboardButton(text="MACD"), types.KeyboardButton(text="SMA"),
     types.KeyboardButton(text="Awesome Oscillator")],
    [types.KeyboardButton(text="ADX"), types.KeyboardButton(text="Williams")],
    [types.KeyboardButton(text="Find Stocks"), types.KeyboardButton(text="Exit")]
]
stock_buttons = [
    [types.KeyboardButton(text="Know more"), types.KeyboardButton(text="Exit")]
]


@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    action = 0
    indicators = list()
    kb = [[types.KeyboardButton(text="📉 Buy"), types.KeyboardButton(text="📈 Sell")]]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    image = ("https://img.freepik.com/premium-vector/robotic-process-automation-abstract-concept-vector"
             "-illustration_107173-25840.jpg?w=826")
    await message.answer_photo(photo=image,
                               caption="💸 Hello ! We are happy to present you MonInvAIBot! That ‘s designed to help "
                                       "you with choosing stocks for investment. \n" +
                                       "💸 How it works:\n" +
                                       "💵 You can use different technical indicators and their combination to "
                                       "filter stocks (at least one should be selected, if after selection of "
                                       "indicators for you strategy you get zero stocks - that means there "
                                       "are no stocks required for all selected indicators and you need to "
                                       "change something in you strategy to get more stocks. \n" +
                                       "💶 We provide forecast of prices movements for 5 days using Long-term "
                                       "Forecasting with TiDE: Time-series Dense Encoder and 50 days plot "
                                       "classification based on CNN, to know more about patterns write /patterns. \n"
                                       "💎 What do you want to do with the shares ? (press one of two buttons) ",
                               reply_markup=keyboard)


# @dp.message_handler(Command("patterns"))
# async def patterns(message: types.Message):
#     kb = [[types.KeyboardButton(text="📉 Buy"), types.KeyboardButton(text="📈 Sell")]]
#     keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
#     image = ("https://img.freepik.com/premium-vector/robotic-process-automation-abstract-concept-vector"
#              "-illustration_107173-25840.jpg?w=826")
#     await message.answer_photo(photo=image,
#                                caption="💸 Hello ! We are happy to present you MonInvAIBot! That ‘s designed to help "
#                                        "you with choosing stocks for investment. \n" +
#                                        "💸 How it works:\n" +
#                                        "💵 You can use different technical indicators and their combination to "
#                                        "filter stocks (at least one should be selected, if after selection of "
#                                        "indicators for you strategy you get zero stocks - that means there "
#                                        "are no stocks required for all selected indicators and you need to "
#                                        "change something in you strategy to get more stocks. \n" +
#                                        "💶 We provide forecast of prices movements for 5 days using Long-term "
#                                        "Forecasting with TiDE: Time-series Dense Encoder and 50 days plot "
#                                        "classification based on CNN, to know more about patterns write /patterns. \n"
#                                        "💎 What do you want to do with the shares ? (press one of two buttons) ",
#                                reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "📈 Sell")
async def sell(message: types.Message):
    actions.append(1)
    print(actions[-1])
    keyboard = types.ReplyKeyboardMarkup(keyboard=indicators_buttons)
    await message.reply('Choose Indicators (you can choose one or several) and press "Find Stocks"',
                        reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "📉 Buy")
async def buy(message: types.Message):
    actions.append(2)
    print(actions[-1])
    keyboard = types.ReplyKeyboardMarkup(keyboard=indicators_buttons)
    await message.reply('Choose indicators for your strategy (you can choose one or several) and press "Find Stocks"',
                        reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "Exit")
async def return_to_start(message: types.Message):
    actions.clear()
    indicators.clear()
    kb = [[types.KeyboardButton(text="📉 Buy"), types.KeyboardButton(text="📈 Sell")]]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    image = ("https://img.freepik.com/premium-vector/robotic-process-automation-abstract-concept-vector"
             "-illustration_107173-25840.jpg?w=826")
    await message.answer_photo(photo=image,
                               caption="💵 You can use different technical indicators and their combination to "
                                       "filter stocks (at least one should be selected, if after selection of "
                                       "indicators for you strategy you get zero stocks - that means there "
                                       "are no stocks required for all selected indicators and you need to "
                                       "change something in you strategy to get more stocks. \n" +
                                       "💶 We provide forecast of prices movements for 5 days using Long-term "
                                       "Forecasting with TiDE: Time-series Dense Encoder and 50 days plot "
                                       "classification based on CNN, to know more about patterns write /patterns. \n"
                                       "💎 What do you want to do with the shares ? (press one of two buttons) ",
                               reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "RSI" or message.text == "Bollinger Band" or
                                    message.text == "Stochastic RSI" or message.text == "MACD" or message.text == "SMA"
                                    or message.text == "Williams" or message.text == "Awesome Oscillator"
                                    or message.text == "ADX")
async def without_puree(message: types.Message):
    indicators.add(message.text)


class Start(StatesGroup):
    start_name = State()


@dp.message_handler(lambda message: message.text == 'Know more', state=None)
async def start_message(message: types.Message):
    await bot.send_message(message.from_user.id, text='Enter stock:')
    await Start.start_name.set()


@dp.message_handler(state=Start.start_name)
async def get_selected_stock_info(message: types.Message):
    res = filter_matrix[filter_matrix.stock == message.text]
    if len(res) > 0:
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
        y = res[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].values.tolist()[0]
        # plt.plot([i for i in range(50)],
        #              list(np.array(filter_matrix['close_scaled'].values[0].data_array()).flatten()[-50:]))
        # plt.plot([i + 50 for i in range(5)], y)
        # plt.savefig(f'img_{message.from_user.id}.png', dpi=100)
        keyboard = types.ReplyKeyboardMarkup(keyboard=stock_buttons)
        await message.answer(send_stocks_details(res), parse_mode=ParseMode.HTML)
        await message.answer('Prise movements for next 5 days: ' + str(y) +
                             "\n 🟥 - price goes down 🟩 - price rises ⬜️ - price doesn't change",
                             parse_mode=ParseMode.HTML, reply_markup=keyboard)
    else:
        actions.clear()
        indicators.clear()
        keyboard = types.ReplyKeyboardMarkup(keyboard=stock_buttons)
        await message.answer("There is no such stock in selected.", reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "Find Stocks")
async def fined_stocks_for_strategy(message: types.Message):
    def select(labelled_matrix, indicators_list, action_type):
        selected_df = labelled_matrix.copy()
        if action_type == 1:
            for indicator in indicators_list:
                if indicator == 'RSI':
                    selected_df = selected_df[labelled_matrix.rsi == 1]
                elif indicator == 'Bollinger Band':
                    selected_df = selected_df[labelled_matrix.bollinger == 1]
                elif indicator == 'Stochastic RSI':
                    selected_df = selected_df[labelled_matrix.stochrsi == 1]
                elif indicator == 'MACD':
                    selected_df = selected_df[labelled_matrix.macd == 1]
                elif indicator == 'SMA':
                    selected_df = selected_df[labelled_matrix.sma == 1]
                elif indicator == 'Williams':
                    selected_df = selected_df[labelled_matrix.williams == 1]
                elif indicator == 'ADX':
                    selected_df = selected_df[labelled_matrix.adx == 1]
                elif indicator == 'Awesome Oscillator':
                    selected_df = selected_df[labelled_matrix.ao == 1]
        if action_type == 2:
            for indicator in indicators_list:
                if indicator == 'RSI':
                    selected_df = selected_df[labelled_matrix.rsi == 2]
                elif indicator == 'Bollinger Band':
                    selected_df = selected_df[labelled_matrix.bollinger == 2]
                elif indicator == 'Stochastic RSI':
                    selected_df = selected_df[labelled_matrix.stochrsi == 2]
                elif indicator == 'MACD':
                    selected_df = selected_df[labelled_matrix.macd == 2]
                elif indicator == 'SMA':
                    selected_df = selected_df[labelled_matrix.sma == 2]
                elif indicator == 'Williams':
                    selected_df = selected_df[labelled_matrix.williams == 2]
                elif indicator == 'ADX':
                    selected_df = selected_df[labelled_matrix.adx == 2]
                elif indicator == 'Awesome Oscillator':
                    selected_df = selected_df[labelled_matrix.ao == 2]
        return selected_df

    if actions[-1] == 1:
        print('Sell')
    else:
        print('Buy')
    res = select(filter_matrix, list(indicators), actions[-1])
    res.sort_values('volume')
    if actions[-1] == 1:
        res = res.sort_values(by=['rises'], ascending=True)
    if actions[-1] == 2:
        res = res.sort_values(by=['rises'], ascending=False)
    res = res[:10] if len(res) >= 10 else res
    indicators.clear()
    if len(res) == 0:
        kb = [[types.KeyboardButton(text="📉 Buy"), types.KeyboardButton(text="📈 Sell")]]
        keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
        await message.answer("Sorry! Technical work is continuing. Our wonderful service will return soon :)",
                             reply_markup=keyboard)
    else:
        keyboard = types.ReplyKeyboardMarkup(keyboard=stock_buttons)
        await message.answer(send_filtered_stocks(res), parse_mode=ParseMode.HTML)
        await message.answer("If you want to know more about some stock - press 'Know more' and enter stock name.",
                             reply_markup=keyboard)


def send_filtered_stocks(selected_stocks):
    table = prettytable.PrettyTable(['Symbol', 'Exchange', 'Close', 'Volume'])
    columns = ['stock', 'exchange', 'close', 'volume']
    for row in selected_stocks[columns].values.tolist():
        table.add_row(row)
    return f'<pre>{table}</pre>'


def send_stocks_details(selected_stocks):
    table = prettytable.PrettyTable(['Item', 'Value'])
    selected_stocks = selected_stocks[['close', 'open', 'high', 'low', 'volume', 'rsi_value', 'stochrsi_value',
                                       'boll_ub_value', 'boll_lb_value', 'macd_value', 'williams_value',
                                       'sma_50_value', 'sma_200_value']].values.tolist()[0]
    print(selected_stocks)
    res = [['Close', round(selected_stocks[0], 3)], ['Open', round(selected_stocks[1], 3)],
           ['High', round(selected_stocks[2], 3)], ['Low', round(selected_stocks[3], 3)],
           ['Volume', round(selected_stocks[4], 3)], ['RSI', round(selected_stocks[5], 3)],
           ['Stochastic RSI', round(selected_stocks[6], 3)], ['Bollinger Band Up', round(selected_stocks[7], 3)],
           ['Bollinger Band Down', round(selected_stocks[8], 3)], ['MACD', round(selected_stocks[9], 3)],
           ['Williams', round(selected_stocks[10], 3)], ['SMA 50', round(selected_stocks[11], 3)],
           ['SMA 200', round(selected_stocks[12], 3)]]
    for row in res:
        table.add_row(row)
    return f'<pre>{table}</pre>'


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("Hello!")


if __name__ == '__main__':
    indicators = set()
    actions = []  # hold
    result = []
    filter_matrix = load()
    print(len(filter_matrix))
    executor.start_polling(dp)
