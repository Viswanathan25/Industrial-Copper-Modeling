import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
import numpy as np
import pickle
import sklearn
from datetime import date
import time

#Setting Page Configuration
st.set_page_config(page_title="Industrial copper modeling | By Viswanathan",layout="wide")

#Creating the option menu in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="ICM",
        options = ['Home', "Classification Model","Regression Model"],
        icons=['house','',''],
        menu_icon='alexa',
        default_index=0

    )

#Into
if selected == 'Home':
      st.title(":violet[*Industrial copper modeling*] By Viswanathan")
      col1, col2 = st.columns(2)
      with col1:
          col1.markdown("## :red[*Domain*] : Copper Manufacturing")
          col1.markdown("## :red[*Technologies used*] : Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit.")
          col1.markdown("## :red[*Overview*] : Build regression model to predict selling price and classification model to predict status")

      with col2:
         col2.markdown("# ")
         col2.image("img.png")
         col2.markdown("#")
         col2.image("img1.jpg")
         col2.markdown("#")


if selected == "Classification Model":
    col1, col2, col3 = st.columns([4, 10, 2])
    with col2:
        st.markdown(
            "<h1 style='font-size: 100px;'><span style='color: cyan;'>Classification </span><span style='color: white;'> Model</span> </h1>",
            unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 10, 5])
    with col2:
        colored_header(
            label="",
            description="",
            color_name="blue-green-70"
        )
    col1, col2, col3 = st.columns([2, 10, 2])
    # Start from options
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")

    with col2:
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Quantity  </span><span style='color: white;'> Ton </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=0.1,max_value=1000000000.0")
        qt = st.number_input('', min_value=0.1, max_value=1000000000.0, value=1.0)
        quantity_log = np.log(qt)

        # ___________________________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Customer  </span><span style='color: white;'> Value </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=12458.0,max_value=2147483647.0")
        customer = st.number_input('', min_value=12458.0, max_value=2147483647.0, value=12458.0, )
        # ________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Country  </span><span style='color: white;'> Code </span> </h1>",
            unsafe_allow_html=True)
        country = st.selectbox(' ', [28, 38, 78, 27, 30, 32, 77, 25, 113, 26, 39, 40, 84, 80, 79, 89, 107])
        # ________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Item  </span><span style='color: white;'> Type </span> </h1>",
            unsafe_allow_html=True)
        cc = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}
        item_type = st.selectbox('          ', cc)

        # ________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Application </span><span style='color: white;'> Code </span> </h1>",
            unsafe_allow_html=True)
        av = st.selectbox('          ', [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0,
                                         27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0,
                                         59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0])

        application_log = np.log(av)
        # ________________________________________________________________________

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Product </span><span style='color: white;'> Referal Code</span> </h1>",
            unsafe_allow_html=True)

        pr = [1670798778, 611993, 1668701376, 164141591, 628377,
              1671863738, 640665, 1332077137, 1668701718, 640405,
              1693867550, 1665572374, 1282007633, 1668701698, 628117,
              1690738206, 640400, 1671876026, 628112, 164336407,
              164337175, 1668701725, 1665572032, 611728, 1721130331,
              1693867563, 611733, 1690738219, 1722207579, 1665584662,
              1665584642, 929423819, 1665584320]
        product_ref = st.selectbox("", pr)

        # ________________________________________________________________________
        with col2:
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Thickness  </span><span style='color: white;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=0.1, max_value=2500.000000")
            thickness = st.number_input('', min_value=0.1, max_value=2500.000000, value=1.0)
            thickness_log = np.log(thickness)
            # ________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Width  </span><span style='color: white;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=1.0, max_value=2990.000000")
            wv = st.number_input('', min_value=1.0, max_value=2990.000000, value=1.0)
            width_log = np.log(wv)

            # ________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Item  </span><span style='color: white;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(1995,1,1),max_Date(2021,12,31)")
            item_date = st.date_input(label='', min_value=date(1995, 1, 1),
                                      max_value=date(2021, 12, 31), value=date(2021, 8, 1))
            # __________________________________________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Delivery </span><span style='color: white;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(2020,1,1),max_date=date(2023,12,31)")
            delivery_date = st.date_input(label='    ', min_value=date(2020, 1, 1),
                                          max_value=date(2023, 12, 31), value=date(2021, 8, 1))
            # ___________________________________________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Selling </span><span style='color: white;'> Price </span> </h1>",
                unsafe_allow_html=True)
            sp = st.number_input('', min_value=1.0, max_value=100001015.0, value=1.0)
            selling_price = np.log(sp)

            predict_data = [quantity_log, customer, country, cc[item_type], application_log, thickness_log, width_log,
                            product_ref, item_date.day,
                            item_date.month, item_date.year, delivery_date.day, delivery_date.month, delivery_date.year,
                            selling_price]

            with open('classification_dataset.pkl', 'rb') as f:
                model = pickle.load(f)
        col1, col2, col3 = st.columns([10, 2, 10])

        with col1:
            st.write("")
            if st.button('Process'):
                x = model.predict([predict_data])
                if x[0] == 1.0:
                    st.markdown(
                        "<h1 style='font-size: 40px;'><span style='color: cyan;'>Predicted Status : </span><span style='color: white;'> Won </span> </h1>",
                        unsafe_allow_html=True)

                elif x[0] == 0.0:
                    st.markdown(
                        "<h1 style='font-size: 40px;'><span style='color: cyan;'>Predicted Status : </span><span style='color: white;'> Lost </span> </h1>",
                        unsafe_allow_html=True)

elif selected == 'Regression Model':
    col1, col2, col3 = st.columns([4, 10, 2])
    with col2:
        st.markdown(
            "<h1 style='font-size: 100px;'><span style='color: cyan;'>Regression </span><span style='color: white;'> Model</span> </h1>",
            unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 10, 7])
    with col2:
        colored_header(
            label="",
            description="",
            color_name="blue-green-70"
        )
    col1, col2, col3 = st.columns([2, 10, 2])
    # Start from options
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")

    with col2:
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Quantity  </span><span style='color: white;'> Ton </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=0.1,max_value=1000000000.0")
        qt = st.number_input('', min_value=0.1, max_value=1000000000.0, value=1.0)
        quantity_log = np.log(qt)

        # ___________________________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Customer  </span><span style='color: white;'> Value </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=12458.0,max_value=2147483647.0")
        customer = st.number_input('', min_value=12458.0, max_value=2147483647.0, value=12458.0, )
        # ________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Country  </span><span style='color: white;'> Code </span> </h1>",
            unsafe_allow_html=True)
        country = st.selectbox(' ', [28, 38, 78, 27, 30, 32, 77, 25, 113, 26, 39, 40, 84, 80, 79, 89, 107])
        # ________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Item  </span><span style='color: white;'> Type </span> </h1>",
            unsafe_allow_html=True)
        cc = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}
        item_type = st.selectbox('          ', cc)

        # ________________________________________________________________________
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Application </span><span style='color: white;'> Code </span> </h1>",
            unsafe_allow_html=True)
        av = st.selectbox('          ', [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0,
                                         27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0,
                                         59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0])

        application_log = np.log(av)
        # ________________________________________________________________________

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: cyan;'>Product </span><span style='color: white;'> Referal Code</span> </h1>",
            unsafe_allow_html=True)

        pr = [1670798778, 611993, 1668701376, 164141591, 628377,
              1671863738, 640665, 1332077137, 1668701718, 640405,
              1693867550, 1665572374, 1282007633, 1668701698, 628117,
              1690738206, 640400, 1671876026, 628112, 164336407,
              164337175, 1668701725, 1665572032, 611728, 1721130331,
              1693867563, 611733, 1690738219, 1722207579, 1665584662,
              1665584642, 929423819, 1665584320]
        product_ref = st.selectbox("", pr)

        # ________________________________________________________________________
        with col2:
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Thickness  </span><span style='color: white;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=0.1, max_value=2500.000000")
            thickness = st.number_input('', min_value=0.1, max_value=2500.000000, value=1.0)
            thickness_log = np.log(thickness)
            # ________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Width  </span><span style='color: white;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=1.0, max_value=2990.000000")
            wv = st.number_input('', min_value=1.0, max_value=2990.000000, value=1.0)
            width_log = np.log(wv)

            # ________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Item  </span><span style='color: white;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(1995,1,1),max_Date(2021,12,31)")
            item_date = st.date_input(label='', min_value=date(1995, 1, 1),
                                      max_value=date(2021, 12, 31), value=date(2021, 8, 1))
            # __________________________________________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Delivery </span><span style='color: white;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(2020,1,1),max_date=date(2023,12,31)")
            delivery_date = st.date_input(label='    ', min_value=date(2020, 1, 1),
                                          max_value=date(2023, 12, 31), value=date(2021, 8, 1))
            # ___________________________________________________________________________________________________________
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: cyan;'>Status </span><span style='color: white;'> Code </span> </h1>",
                unsafe_allow_html=True)
            status_code = {'Won': 1, 'Draft': 2, 'To be approved': 3, 'Lost': 0, 'Not lost for AM': 5, 'Wonderful': 6,
                           'Revised': 7, 'Offered': 8, 'Offerable': 4}
            Status = st.selectbox('             ', status_code)

            # _____________________________________________________________________________________________________________

            predict_data = [quantity_log, customer, country, cc[item_type], application_log, thickness_log, width_log,
                            product_ref, item_date.day,
                            item_date.month, item_date.year, delivery_date.day, delivery_date.month, delivery_date.year,
                            status_code[Status]]

            with open('regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
        col1, col2, col3 = st.columns([10, 1, 10])

        with col1:
            st.write("")
            if st.button('Process'):
                x = model.predict([predict_data])
                st.markdown(
                    f"<h1 style='font-size: 40px;'><span style='color: cyan;'>Predicted Selling Price : </span><span style='color: white;'> {np.exp(x[0])}</span> </h1>",
                    unsafe_allow_html=True)


#===========================================================================END================================================================================================#