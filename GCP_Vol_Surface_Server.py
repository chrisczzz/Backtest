from GCP_Option_Hedging_Toolkit import *
from xmlrpc.server import SimpleXMLRPCServer
def set_parmas():
    global ATM_csv_path,C_25D_csv_path,P_25D_csv_path,C_10D_csv_path,P_10D_csv_path,contract_expiry_days,trade_close_period,port_number
    ATM_csv_path = sys.argv[1]
    C_25D_csv_path = sys.argv[2]
    P_25D_csv_path = sys.argv[3]
    C_10D_csv_path = sys.argv[4]
    P_10D_csv_path = sys.argv[5]
    contract_expiry_days=int(sys.argv[6])
    trade_close_period=int(sys.argv[7])
    port_number=int(sys.argv[8])

def prepare_vol_surface(ATM_csv_path,C_25D_csv_path,P_25D_csv_path,C_10D_csv_path,P_10D_csv_path,contract_expiry_days,trade_close_period):
    ATM_vol = pd.read_csv(ATM_csv_path, header=0, index_col=[0], parse_dates=[0])
    C_25D_vol = pd.read_csv(C_25D_csv_path, header=0, index_col=[0], parse_dates=[0])
    P_25D_vol = pd.read_csv(P_25D_csv_path, header=0, index_col=[0], parse_dates=[0])
    C_10D_vol = pd.read_csv(C_10D_csv_path, header=0, index_col=[0], parse_dates=[0])
    P_10D_vol = pd.read_csv(P_10D_csv_path, header=0, index_col=[0], parse_dates=[0])

    print("First day of vol surface: ",ATM_vol.index[0])
    print("Last day of vol surface: ",ATM_vol.index[-1])
    print("First day of tenor: ",(contract_expiry_days-trade_close_period))
    print("Last day of tenor: ", contract_expiry_days)
    smile_dict = {}
    for i in range(len(ATM_vol.index)):
        print ("Generating vol surface for date = ",ATM_vol.index[i])
        smile_dict_thisday = {}
        for j in range(contract_expiry_days-trade_close_period,contract_expiry_days+1):
            atm_point=ATM_vol.loc[ATM_vol.index[i]][j]
            c_point_1 = C_25D_vol.loc[C_25D_vol.index[i]][j]
            c_point_2 = C_10D_vol.loc[C_10D_vol.index[i]][j]
            p_point_1 = P_25D_vol.loc[P_25D_vol.index[i]][j]
            p_point_2 = P_10D_vol.loc[P_10D_vol.index[i]][j]
            smile_dict_thisday[j] = sci.interpolate.CubicSpline([0.1,0.25,0.5,0.75,0.9], [p_point_2,p_point_1,atm_point,c_point_1,c_point_2], axis=1, bc_type='not-a-knot', extrapolate=True)
        smile_dict[ATM_vol.index[i]] = smile_dict_thisday
    global vol_surface
    vol_surface = smile_dict
def get_vol_surface():
    #return vol_surface
    return  pickle.dumps(vol_surface)
if __name__ == '__main__':
    set_parmas()
    prepare_vol_surface(ATM_csv_path, C_25D_csv_path, P_25D_csv_path, C_10D_csv_path, P_10D_csv_path,
                            contract_expiry_days, trade_close_period)
    server = SimpleXMLRPCServer(("localhost", port_number))
    print("Listening on port: ",port_number)
    server.register_function(get_vol_surface, "get_vol_surface")
    server.serve_forever()