from pytorch3d.transforms import quaternion_to_matrix
import torch

from rotation_learning import A_vec_to_quat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_yhat_to_transforms(y_hat):
    lhand_pos_hat = y_hat[:, 0:3]
    rhand_pos_hat = y_hat[:, 13:16]

    lhand_A_vec = y_hat[:, 3:13] #Bx10
    rhand_A_vec = y_hat[:, 16:26] #Bx10
    lhand_quat_hat = A_vec_to_quat(lhand_A_vec) #Bx10 -> Bx4
    rhand_quat_hat = A_vec_to_quat(rhand_A_vec) #Bx10 -> Bx4
    lhand_rot_hat = quaternion_to_matrix(lhand_quat_hat)
    rhand_rot_hat = quaternion_to_matrix(rhand_quat_hat)

    T_lhand_as_3x4 = torch.cat([lhand_rot_hat, lhand_pos_hat.unsqueeze(dim=2)], dim=2)
    T_rhand_as_3x4 = torch.cat([rhand_rot_hat, rhand_pos_hat.unsqueeze(dim=2)], dim=2)

    last_row = torch.zeros((y_hat.size()[0], 1, 4)).to(lhand_pos_hat.device)
    last_row[:, :, 3] = 1

    T_cam_lhandHat = torch.cat([T_lhand_as_3x4, last_row], dim=1)
    T_cam_rhandHat = torch.cat([T_rhand_as_3x4, last_row], dim=1)

    return T_cam_lhandHat, T_cam_rhandHat


def position_loss(y, y_hat):
    lhand_pos = y[:, 0:3]
    rhand_pos = y[:, 7:10]
    lhand_pos_hat = y_hat[:, 0:3]
    rhand_pos_hat = y_hat[:, 13:16]

    pos = torch.cat([lhand_pos, rhand_pos], dim=1)
    pos_hat = torch.cat([lhand_pos_hat, rhand_pos_hat], dim=1)

    lhand_pos_error = pos[:, :3] - pos_hat[:, :3]
    rhand_pos_error = pos[:, 3:] - pos_hat[:, 3:]

    avg_lhand_pos_error = torch.linalg.norm(lhand_pos_error, dim=1).mean().detach().item()
    avg_lhand_xyz_error = torch.abs(torch.mean(lhand_pos_error, dim=0)).detach().cpu().numpy()
    avg_rhand_pos_error = torch.linalg.norm(rhand_pos_error, dim=1).mean().detach().item()
    avg_rhand_xyz_error = torch.abs(torch.mean(rhand_pos_error, dim=0)).detach().cpu().numpy()
    avg_pos_error = (avg_rhand_pos_error + avg_lhand_pos_error) / 2.0
    avg_xyz_error = (avg_lhand_xyz_error + avg_rhand_xyz_error) / 2.0
    info = {
        "avg_lhand_pos_error": avg_lhand_pos_error,
        "avg_rhand_pos_error": avg_rhand_pos_error,
        "avg_hand_pos_error": avg_pos_error,
        "avg_lhand_xyz_error": avg_lhand_xyz_error,
        "avg_rhand_xyz_error": avg_rhand_xyz_error,
        "avg_xyz_error": avg_xyz_error,
    }

    mse_loss = torch.nn.MSELoss()
    return mse_loss(pos, pos_hat), info


def rotation_loss(y, y_hat):
    lhand_quat = y[:, 3:7]
    rhand_quat = y[:, 10:14]

    # rotation magic
    lhand_A_vec = y_hat[:, 3:13] #Bx10
    rhand_A_vec = y_hat[:, 16:26] #Bx10
    lhand_quat_hat = A_vec_to_quat(lhand_A_vec) #Bx10 -> Bx4
    rhand_quat_hat = A_vec_to_quat(rhand_A_vec) #Bx10 -> Bx4

    quat = torch.cat([lhand_quat, rhand_quat], dim=1)
    quat_hat = torch.cat([lhand_quat_hat, rhand_quat_hat], dim=1)

    # quat difference loss (eqn 16/20 from paper)
    lhand_loss = torch.min((lhand_quat-lhand_quat_hat).norm(dim=1), (lhand_quat+lhand_quat_hat).norm(dim=1)).squeeze()
    rhand_loss = torch.min((rhand_quat-rhand_quat_hat).norm(dim=1), (rhand_quat+rhand_quat_hat).norm(dim=1)).squeeze()

    info = {}

    return torch.mean(lhand_loss + rhand_loss) / 2.0, info
