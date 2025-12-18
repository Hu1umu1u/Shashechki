#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <unordered_map>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <cmath>
#include <cfloat>

const unsigned int win_w = 1920;
const unsigned int win_h = 1080;
float car_scale = 5.0f;
float car_y_off = 0.2f;
float track_half_len = 2000.0f;
float track_half_w   = 20.0f;
const int   max_pl     = 16;
const float lamp_y     = 6.0f;
const float lamp_step  = 40.0f;

std::vector<glm::vec3> lamp_pos;

// фоновый свет
glm::vec3 amb_col  = glm::vec3(0.03f, 0.04f, 0.08f);
glm::vec3 dir_dir  = glm::vec3(-0.3f, -1.0f, -0.2f);
float     dir_int  = 0.35f;

// ближний
glm::vec3 hl_col      = glm::vec3(3.0f, 2.6f, 2.2f);
float     hl_rad      = 140.0f;
float     hl_ang_deg  = 36.0f;
float     hl_y        = 0.4f;
float     hl_fwd      = 2.3f;
float     hl_side     = 0.7f;

// стопы
glm::vec3 tail_col_idle   = glm::vec3(1.2f, 0.18f, 0.06f);
glm::vec3 tail_col_brake  = glm::vec3(3.5f, 0.6f, 0.18f);
float     tail_rad_idle   = 20.0f;
float     tail_rad_brake  = 80.0f;
float     tail_ang_deg    = 30.0f;
float     tail_y          = 0.2f;
float     tail_back       = 0.6f;
float     tail_side       = 0.7f;

// фонари
glm::vec3 lamp_col    = glm::vec3(1.6f, 1.3f, 0.8f);
float     lamp_rad    = 100.0f;
float     lamp_ang_deg= 60.0f;

// настройки шлагбаум
bool gate_open = false;
float gate_x = 0.0f;
float gate_z = 0.0f;
float gate_half_width = 3.0f;
float gate_solid_half = 2.5f;
float gate_stop_dist = 10.0f;
float gate_angle = 0.0f;
const float gate_angle_open = glm::radians(75.0f);
const float gate_anim_speed = 4.0f;

// камера
float cam_dist      = 6.0f;
float cam_height    = 3.0f;
float cam_spring    = 10.0f;
float cam_target_h  = 0.5f;
bool mouse_down = false;
double last_mouse_x = 0.0, last_mouse_y = 0.0;
float orbit_az = 0.0f;
float orbit_el = glm::radians(20.0f);
float orbit_dist = cam_dist;
bool orbit_mode = false;
float mouse_sens = 0.002f;

// настройки
float orbit_min_el = glm::radians(-5.0f);
float orbit_max_el = glm::radians(89.0f);
float orbit_min_dist = 5.0f;
float orbit_max_dist = 20.0f;
float orbit_zoom_step = 0.90f;
float min_cam_height = 0.15f;

// отбойник настройки
const float rail_seg_len = 30.0f;
const float guard_x_factor = 0.9f;

glm::vec3 prev_car_pos = glm::vec3(0.0f);

struct vertex {
    glm::vec3 pos;
    glm::vec3 nrm;
    glm::vec2 uv;
};

struct mesh {
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;
    unsigned int idx_count = 0;
};

struct material_info {
    std::string name;
    glm::vec3 kd = glm::vec3(1.0f);
    std::string map_kd;
    unsigned int tex = 0;
};

struct submesh {
    std::string name;
    mesh m;
    unsigned int tex = 0;
    glm::vec3 kd = glm::vec3(1.0f);
    float spin = 0.0f;
    glm::vec3 origin = glm::vec3(0.0f);
};

struct obj_model {
    mesh m;
    unsigned int tex = 0;
    std::vector<submesh> parts;
};

struct car_state {
    glm::vec3 pos = glm::vec3(0.0f, 0.5f, 0.0f);
    glm::vec3 vel = glm::vec3(0.0f);
    float yaw   = 0.0f;
    float steer = 0.0f;
    bool  brake = false;
};

struct cam_state {
    glm::vec3 pos;
    glm::vec3 target;
    glm::vec3 up;
};

struct traffic_car {
    glm::vec3 pos;
    float dir;
    float speed;
    float max_speed;
    glm::vec3 color;
    bool stopped;
    float stop_timer;
    float mass;
};

const float TRAFFIC_STOP_BUFFER_DIST = 6.0f;
const float TRAFFIC_SLOW_SPEED_THRESH = 0.5f;


GLFWwindow* win = nullptr;

float dt = 0.0f;
float last_time = 0.0f;

car_state car;
cam_state cam;

obj_model car_model;
obj_model track_model;
mesh      lamp_mesh;
mesh gate_mesh;

obj_model tree_trunk_model;
obj_model tree_leaves_model;
obj_model bench_wood_model;
obj_model bench_metal_model;
obj_model bush_model;
obj_model rail_model;
obj_model grass_model;
float tree_scale  = 3.5f;
float bush_scale  = 2.0f;
float bench_scale = 2.0f;
std::vector<glm::vec3> tree_pos;
std::vector<glm::vec3> bench_pos;
std::vector<glm::vec3> bush_pos;

obj_model traffic_car_model;
mesh      light_sphere_mesh;
std::vector<traffic_car> traffic;

unsigned int prog = 0;

bool key_w = false;
bool key_s = false;
bool key_a = false;
bool key_d = false;
bool key_r = false;
bool key_esc = false;
bool key_space = false;

const char* vs_src = R"(#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_nrm;
layout (location = 2) in vec2 a_uv;

out vec3 v_nrm;
out vec2 v_uv;
out vec3 v_world_pos;

uniform mat4 u_m;
uniform mat4 u_v;
uniform mat4 u_p;

void main() {
    vec4 world = u_m * vec4(a_pos, 1.0);
    v_world_pos = world.xyz;
    v_nrm = mat3(transpose(inverse(u_m))) * a_nrm;
    v_uv = a_uv;
    gl_Position = u_p * u_v * world;
}
)";

const char* fs_src = R"(#version 460 core
in vec3 v_nrm;
in vec2 v_uv;
in vec3 v_world_pos;

out vec4 frag;

uniform sampler2D u_tex;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;

// фоновый свет
uniform vec3  u_amb;
uniform float u_dir_int;
uniform vec3  u_emis;
uniform vec3  u_mat_col;

// точечный свет
const int MAX_PL = 16;
uniform int   u_pl_count;
uniform vec3  u_pl_pos[MAX_PL];
uniform vec3  u_pl_col[MAX_PL];
uniform float u_pl_rad[MAX_PL];
uniform vec3  u_pl_dir[MAX_PL];
uniform float u_pl_cos[MAX_PL];

void main() {
    vec3 n = normalize(v_nrm);
    vec3 view_dir = normalize(u_cam_pos - v_world_pos);

    vec3 albedo = texture(u_tex, v_uv).rgb;
    vec3 base   = albedo * u_mat_col;

    vec3 col = base * u_amb;

    // фоновый свет
    if (u_dir_int > 0.0) {
        vec3 l = normalize(-u_light_dir);
        float diff = max(dot(n, l), 0.0);
        vec3 refl = reflect(-l, n);
        float spec = pow(max(dot(view_dir, refl), 0.0), 32.0);
        col += base * diff * u_dir_int;
        col += vec3(1.0) * spec * u_dir_int * 0.3;
    }

    // точечный свет
    for (int i = 0; i < u_pl_count; ++i) {
        vec3 to_light = u_pl_pos[i] - v_world_pos;
        float dist = length(to_light);
        if (dist > u_pl_rad[i]) continue;

        vec3 l = to_light / dist;
        float att = 1.0 - dist / u_pl_rad[i];

        float spot = 1.0;
        if (u_pl_cos[i] > 0.0) {
            vec3 light_to_frag = -l;
            float cosTheta = dot(normalize(light_to_frag), normalize(u_pl_dir[i]));
            if (cosTheta < u_pl_cos[i]) continue;

            float x = (cosTheta - u_pl_cos[i]) / (1.0 - u_pl_cos[i]);
            x = clamp(x, 0.0, 1.0);
            spot = x * x;
        }

        float diff_pl = max(dot(n, l), 0.0);
        vec3 half_dir = normalize(l + view_dir);
        float spec_pl = pow(max(dot(n, half_dir), 0.0), 32.0);

        vec3 light_col = u_pl_col[i];

        col += base * diff_pl * light_col * att * spot;
        col += light_col * spec_pl * att * spot * 0.6;
    }

    col += u_emis;

    frag = vec4(col, 1.0);
}
)";

unsigned int make_shader(GLenum type, const char* src) {
    unsigned int s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    int ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, 1024, nullptr, log);
        std::cerr << "[err] shader compile:\n" << log << std::endl;
    }
    return s;
}

unsigned int make_prog(const char* vs, const char* fs) {
    unsigned int vs_id = make_shader(GL_VERTEX_SHADER, vs);
    unsigned int fs_id = make_shader(GL_FRAGMENT_SHADER, fs);

    unsigned int p = glCreateProgram();
    glAttachShader(p, vs_id);
    glAttachShader(p, fs_id);
    glLinkProgram(p);

    int ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, 1024, nullptr, log);
        std::cerr << "[err] program link:\n" << log << std::endl;
    }

    glDeleteShader(vs_id);
    glDeleteShader(fs_id);

    return p;
}

// загрузка текстуры
unsigned int load_tex(const std::string& path) {
    std::cout << "[tex] load " << path << std::endl;

    int w = 0, h = 0, ch = 0;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!data) {
        std::cerr << "[tex] failed " << path << std::endl;
        return 0;
    }

    unsigned int tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
    return tex;
}

// разбор mtl
std::unordered_map<std::string, material_info> parse_mtl(const std::string& path) {
    std::unordered_map<std::string, material_info> mats;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[mtl] can't open: " << path << std::endl;
        return mats;
    }

    material_info cur;
    bool have = false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        std::string tok;
        ss >> tok;
        if (tok == "newmtl") {
            if (have) mats[cur.name] = cur;
            cur = material_info();
            ss >> cur.name;
            have = true;
        } else if (tok == "Kd") {
            ss >> cur.kd.r >> cur.kd.g >> cur.kd.b;
        } else if (tok == "map_Kd") {
            ss >> cur.map_kd;
        }
    }
    if (have) mats[cur.name] = cur;
    return mats;
}

// обновление шлагбаума
void upd_gate(float dt) {
    float target = gate_open ? gate_angle_open : 0.0f;
    float t = glm::clamp(dt * gate_anim_speed, 0.0f, 1.0f);
    gate_angle = glm::mix(gate_angle, target, t);
}

// загрузка модели с mtl
bool load_obj_with_mtl(const std::string& path, obj_model& out_model) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "[err] open obj: " << path << std::endl; return false; }

    // base dir for resolving texture paths
    auto slash = path.find_last_of("/\\");
    std::string base_dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash+1);

    std::vector<glm::vec3> tmp_pos;
    std::vector<glm::vec3> tmp_nrm;
    std::vector<glm::vec2> tmp_uv;

    struct idx { int v, t, n; };

    std::string cur_mat = "default";
    std::map<std::string, std::vector<std::vector<idx>>> faces_by_mat;
    std::string mtl_file;
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;
        if (type == "v") {
            glm::vec3 p; ss >> p.x >> p.y >> p.z; tmp_pos.push_back(p);
        } else if (type == "vt") {
            glm::vec2 uv; ss >> uv.x >> uv.y; tmp_uv.push_back(uv);
        } else if (type == "vn") {
            glm::vec3 n; ss >> n.x >> n.y >> n.z; tmp_nrm.push_back(n);
        } else if (type == "f") {
            std::vector<idx> face;
            std::string tok;
            while (ss >> tok) {
                if (tok.empty()) break;
                idx id{ -1, -1, -1 };

                size_t p1 = tok.find('/');
                size_t p2 = (p1==std::string::npos) ? std::string::npos : tok.find('/', p1 + 1);

                if (p1 == std::string::npos) {
                    id.v = std::stoi(tok);
                } else {
                    std::string sv = tok.substr(0, p1);
                    std::string st = (p2 != std::string::npos && p2 > p1 + 1)
                                     ? tok.substr(p1 + 1, p2 - p1 - 1) : "";
                    std::string sn = (p2 != std::string::npos && p2 + 1 < tok.size())
                                     ? tok.substr(p2 + 1) : "";

                    if (!sv.empty()) id.v = std::stoi(sv);
                    if (!st.empty()) id.t = std::stoi(st);
                    if (!sn.empty()) id.n = std::stoi(sn);
                }
                face.push_back(id);
            }
            if (face.size() >= 3) faces_by_mat[cur_mat].push_back(face);
        } else if (type == "usemtl") {
            ss >> cur_mat;
        } else if (type == "mtllib") {
            ss >> mtl_file;
        }
    }

    if (tmp_pos.empty()) { std::cerr << "[obj] no verts in " << path << std::endl; return false; }

    glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
    for (auto& p : tmp_pos) {
        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
    }
    glm::vec3 size = bmax - bmin;
    glm::vec3 center = (bmin + bmax) * 0.5f;
    float max_dim = std::max(size.x, std::max(size.y, size.z));
    float norm = (max_dim > 0.0f) ? (1.0f / max_dim) : 1.0f;
    for (auto& p : tmp_pos) p = (p - center) * norm;

    std::unordered_map<std::string, material_info> mats;
    if (!mtl_file.empty()) {
        std::string mtl_path = base_dir + mtl_file;
        mats = parse_mtl(mtl_path);
    }

    auto fix_index = [](int idx, int size)->int {
        if (idx == -1) return -1;
        if (idx < 0) return size + idx;
        return idx - 1;
    };

    for (auto &kv : faces_by_mat) {
        const std::string& matname = kv.first;
        const auto& faces = kv.second;

        std::map<std::tuple<int,int,int>, unsigned int> unique;
        std::vector<vertex> verts;
        std::vector<unsigned int> idxs;
        verts.reserve(faces.size() * 3);
        idxs.reserve(faces.size() * 3);

        for (const auto &face : faces) {
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                idx tri[3] = { face[0], face[i], face[i+1] };
                for (int k = 0; k < 3; ++k) {
                    int vi = fix_index(tri[k].v, (int)tmp_pos.size());
                    int ti = fix_index(tri[k].t, (int)tmp_uv.size());
                    int ni = fix_index(tri[k].n, (int)tmp_nrm.size());

                    std::tuple<int,int,int> key(vi, ti, ni);
                    auto it = unique.find(key);
                    if (it == unique.end()) {
                        vertex v{};
                        v.pos = (vi >= 0 && vi < (int)tmp_pos.size()) ? tmp_pos[vi] : glm::vec3(0.0f);
                        if (ti >= 0 && ti < (int)tmp_uv.size())
                            v.uv = glm::vec2(tmp_uv[ti].x, 1.0f - tmp_uv[ti].y);
                        else
                            v.uv = glm::vec2(0.0f);
                        v.nrm = (ni >= 0 && ni < (int)tmp_nrm.size()) ? tmp_nrm[ni] : glm::vec3(0.0f, 1.0f, 0.0f);

                        unsigned int new_idx = (unsigned int)verts.size();
                        verts.push_back(v);
                        unique[key] = new_idx;
                        idxs.push_back(new_idx);
                    } else {
                        idxs.push_back(it->second);
                    }
                }
            }
        }

        if (verts.empty() || idxs.empty()) continue;

        mesh mm;
        glGenVertexArrays(1, &mm.vao);
        glGenBuffers(1, &mm.vbo);
        glGenBuffers(1, &mm.ebo);

        glBindVertexArray(mm.vao);
        glBindBuffer(GL_ARRAY_BUFFER, mm.vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mm.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));
        glBindVertexArray(0);

        mm.idx_count = (unsigned int)idxs.size();

        submesh part;
        part.name = matname;
        part.m = mm;
        part.kd = glm::vec3(1.0f);
        part.tex = 0;
        part.spin = 0.0f;

        glm::vec3 vmin(FLT_MAX), vmax(-FLT_MAX);
        for (const auto &vv : verts) {
            vmin = glm::min(vmin, vv.pos);
            vmax = glm::max(vmax, vv.pos);
        }
        part.origin = (vmin + vmax) * 0.5f;

        if (mats.count(matname)) {
            part.kd = mats[matname].kd;
            std::string tex_rel = mats[matname].map_kd;
            if (!tex_rel.empty()) {
                std::string tex_path = base_dir + tex_rel;
                std::cout << "[mtl->tex] mat '" << matname << "' map_Kd='" << tex_rel << "' -> trying '" << tex_path << "'" << std::endl;
                unsigned int texid = load_tex(tex_path);
                std::cout << "[mtl->tex] result: texid=" << texid << " for mat '" << matname << "'" << std::endl;
                part.tex = texid;
                part.spin = 0.0f;
            } else {
                std::cout << "[mtl->tex] mat '" << matname << "' has no map_Kd" << std::endl;
            }
        } else {
            std::cout << "[mtl->tex] no material entry for '" << matname << "' in MTL" << std::endl;
        }

        out_model.parts.push_back(part);
        std::cout << "[obj] added part name='" << matname << "' tex=" << part.tex << " kd=" << part.kd.r << "," << part.kd.g << "," << part.kd.b << " verts=" << verts.size() << " idxs=" << idxs.size() << std::endl;
    }
    return !out_model.parts.empty();
}

// дорога текстура
unsigned int make_road_texture() {
    const int W = 256;
    const int H = 256;
    std::vector<unsigned char> data(W * H * 4);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = (x + 0.5f) / (float)W;

            // цвет асфальта
            float noise = float((x ^ y) & 7) / 7.0f;
            float base = 0.12f + 0.02f * noise;
            float r = base;
            float g = base;
            float b = base;

            // двойная сплошная жёлтая
            bool center_line =
                (u > 0.49f && u < 0.495f) ||
                (u > 0.505f && u < 0.51f);

            if (center_line) {
                r = 0.95f;
                g = 0.85f;
                b = 0.25f;
            }

            int idx = (y * W + x) * 4;
            data[idx + 0] = (unsigned char)(glm::clamp(r, 0.0f, 1.0f) * 255.0f);
            data[idx + 1] = (unsigned char)(glm::clamp(g, 0.0f, 1.0f) * 255.0f);
            data[idx + 2] = (unsigned char)(glm::clamp(b, 0.0f, 1.0f) * 255.0f);
            data[idx + 3] = 255;
        }
    }

    unsigned int tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    std::cout << "[tex] road texture generated\n";
    return tex;
}


// текстура для трафика
unsigned int make_traffic_car_texture() {
    const int W = 64;
    const int H = 64;
    std::vector<unsigned char> data(W * H * 4);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = (x + 0.5f) / (float)W;
            float v = (y + 0.5f) / (float)H;

            glm::vec3 col(0.7f, 0.7f, 0.7f); // кузов

            // крыша
            if (v > 0.4f && v < 0.9f && u > 0.2f && u < 0.8f) {
                col = glm::vec3(0.25f, 0.27f, 0.30f);
            }

            // лобаш
            if (v < 0.35f && v > 0.15f && u > 0.2f && u < 0.8f) {
                col = glm::vec3(0.15f, 0.20f, 0.30f);
            }

            int idx = (y * W + x) * 4;
            data[idx + 0] = (unsigned char)(glm::clamp(col.r, 0.0f, 1.0f) * 255.0f);
            data[idx + 1] = (unsigned char)(glm::clamp(col.g, 0.0f, 1.0f) * 255.0f);
            data[idx + 2] = (unsigned char)(glm::clamp(col.b, 0.0f, 1.0f) * 255.0f);
            data[idx + 3] = 255;
        }
    }

    unsigned int tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    std::cout << "[tex] traffic car texture generated\n";
    return tex;
}

// загрузка модели
bool load_obj(const std::string& path, mesh& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[err] open obj: " << path << std::endl;
        return false;
    }

    std::vector<glm::vec3> tmp_pos;
    std::vector<glm::vec3> tmp_nrm;
    std::vector<glm::vec2> tmp_uv;

    struct idx { int v, t, n; };
    std::vector<std::vector<idx>> faces;

    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;
        if (type == "v") {
            glm::vec3 p;
            ss >> p.x >> p.y >> p.z;
            tmp_pos.push_back(p);
        } else if (type == "vt") {
            glm::vec2 uv;
            ss >> uv.x >> uv.y;
            tmp_uv.push_back(uv);
        } else if (type == "vn") {
            glm::vec3 n;
            ss >> n.x >> n.y >> n.z;
            tmp_nrm.push_back(n);
        } else if (type == "f") {
            std::vector<idx> face;
            std::string tok;
            while (ss >> tok) {
                if (tok.empty()) break;
                idx id{ -1, -1, -1 };

                size_t p1 = tok.find('/');
                size_t p2 = (p1==std::string::npos) ? std::string::npos : tok.find('/', p1 + 1);

                if (p1 == std::string::npos) {
                    id.v = std::stoi(tok);
                } else {
                    std::string sv = tok.substr(0, p1);
                    std::string st = (p2 != std::string::npos && p2 > p1 + 1)
                                     ? tok.substr(p1 + 1, p2 - p1 - 1) : "";
                    std::string sn = (p2 != std::string::npos && p2 + 1 < tok.size())
                                     ? tok.substr(p2 + 1) : "";

                    if (!sv.empty()) id.v = std::stoi(sv);
                    if (!st.empty()) id.t = std::stoi(st);
                    if (!sn.empty()) id.n = std::stoi(sn);
                }

                face.push_back(id);
            }

            if (face.size() >= 3) faces.push_back(face);
        }
    }

    if (tmp_pos.empty()) {
        std::cerr << "[obj] no verts in " << path << std::endl;
        return false;
    }

    glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
    for (auto& p : tmp_pos) {
        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
    }
    glm::vec3 size = bmax - bmin;
    glm::vec3 center = (bmin + bmax) * 0.5f;
    float max_dim = std::max(size.x, std::max(size.y, size.z));
    float norm = (max_dim > 0.0f) ? (1.0f / max_dim) : 1.0f;

    for (auto& p : tmp_pos) p = (p - center) * norm;

    std::map<std::tuple<int,int,int>, unsigned int> unique;
    std::vector<vertex> verts;
    std::vector<unsigned int> idxs;
    verts.reserve(faces.size() * 3);
    idxs.reserve(faces.size() * 3);

    auto fix_index = [](int idx, int size)->int {
        if (idx == -1) return -1;
        if (idx < 0) return size + idx;
        return idx - 1;
    };

    for (const auto& face : faces) {
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            idx tri[3] = { face[0], face[i], face[i+1] };
            for (int k = 0; k < 3; ++k) {
                int vi = fix_index(tri[k].v, (int)tmp_pos.size());
                int ti = fix_index(tri[k].t, (int)tmp_uv.size());
                int ni = fix_index(tri[k].n, (int)tmp_nrm.size());

                std::tuple<int,int,int> key(vi, ti, ni);
                auto it = unique.find(key);
                if (it == unique.end()) {
                    vertex v{};
                    v.pos = (vi >= 0 && vi < (int)tmp_pos.size()) ? tmp_pos[vi] : glm::vec3(0.0f);
                    v.uv  = (ti >= 0 && ti < (int)tmp_uv.size()) ? tmp_uv[ti] : glm::vec2(0.0f);
                    v.nrm = (ni >= 0 && ni < (int)tmp_nrm.size()) ? tmp_nrm[ni] : glm::vec3(0.0f, 1.0f, 0.0f);

                    unsigned int new_idx = (unsigned int)verts.size();
                    verts.push_back(v);
                    unique[key] = new_idx;
                    idxs.push_back(new_idx);
                } else {
                    idxs.push_back(it->second);
                }
            }
        }
    }

    if (verts.empty() || idxs.empty()) {
        std::cerr << "[obj] no faces produced for " << path << std::endl;
        return false;
    }
    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    glBindVertexArray(0);

    out.idx_count = (unsigned int)idxs.size();
    return true;
}

// куб ген
void add_box(std::vector<vertex>& verts,
             std::vector<unsigned int>& idxs,
             const glm::vec3& bmin,
             const glm::vec3& bmax)
{
    struct face_desc {
        glm::vec3 n;
        glm::vec3 v[4];
    };

    float x0 = bmin.x, x1 = bmax.x;
    float y0 = bmin.y, y1 = bmax.y;
    float z0 = bmin.z, z1 = bmax.z;

    face_desc faces[6] = {
        { glm::vec3( 1,0,0), { {x1,y0,z0},{x1,y0,z1},{x1,y1,z1},{x1,y1,z0} } },
        { glm::vec3(-1,0,0), { {x0,y0,z1},{x0,y0,z0},{x0,y1,z0},{x0,y1,z1} } },
        { glm::vec3(0, 1,0), { {x0,y1,z0},{x1,y1,z0},{x1,y1,z1},{x0,y1,z1} } },
        { glm::vec3(0,-1,0), { {x0,y0,z1},{x1,y0,z1},{x1,y0,z0},{x0,y0,z0} } },
        { glm::vec3(0,0, 1), { {x0,y0,z1},{x0,y1,z1},{x1,y1,z1},{x1,y0,z1} } },
        { glm::vec3(0,0,-1), { {x1,y0,z0},{x1,y1,z0},{x0,y1,z0},{x0,y0,z0} } }
    };

    for (int f = 0; f < 6; ++f) {
        unsigned int base = (unsigned int)verts.size();
        for (int i = 0; i < 4; ++i) {
            vertex v{};
            v.pos = faces[f].v[i];
            v.nrm = faces[f].n;
            v.uv  = glm::vec2((i == 1 || i == 2) ? 1.0f : 0.0f,
                              (i >= 2) ? 1.0f : 0.0f);
            verts.push_back(v);
        }
        idxs.push_back(base + 0);
        idxs.push_back(base + 1);
        idxs.push_back(base + 2);
        idxs.push_back(base + 0);
        idxs.push_back(base + 2);
        idxs.push_back(base + 3);
    }
}

// сфера ген
void add_sphere(std::vector<vertex>& verts,
                std::vector<unsigned int>& idxs,
                const glm::vec3& center,
                float radius,
                int stacks,
                int slices)
{
    unsigned int base = (unsigned int)verts.size();

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float theta = v * glm::pi<float>();

        float sin_t = std::sin(theta);
        float cos_t = std::cos(theta);

        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float phi = u * glm::two_pi<float>();

            float sin_p = std::sin(phi);
            float cos_p = std::cos(phi);

            glm::vec3 n(sin_t * cos_p, cos_t, sin_t * sin_p);
            glm::vec3 p = center + radius * n;

            vertex vert{};
            vert.pos = p;
            vert.nrm = glm::normalize(n);
            vert.uv  = glm::vec2(u, v);
            verts.push_back(vert);
        }
    }

    int ring = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            unsigned int i0 = base + i * ring + j;
            unsigned int i1 = i0 + ring;
            unsigned int i2 = i0 + 1;
            unsigned int i3 = i1 + 1;

            idxs.push_back(i0);
            idxs.push_back(i1);
            idxs.push_back(i2);

            idxs.push_back(i2);
            idxs.push_back(i1);
            idxs.push_back(i3);
        }
    }
}

// дорога
void make_track_mesh(mesh& out) {
    std::vector<vertex> verts(4);
    std::vector<unsigned int> idxs = {0,1,2,0,2,3};

    float half_len = track_half_len;
    float half_w   = track_half_w;

    verts[0].pos = glm::vec3(-half_w, 0.0f, -half_len);
    verts[1].pos = glm::vec3(-half_w, 0.0f,  half_len);
    verts[2].pos = glm::vec3( half_w, 0.0f,  half_len);
    verts[3].pos = glm::vec3( half_w, 0.0f, -half_len);

    for (int i = 0; i < 4; ++i)
        verts[i].nrm = glm::vec3(0,1,0);

    float tile_u = 1.0f;
    float tile_v = 80.0f;

    verts[0].uv = glm::vec2(0.0f,    0.0f);
    verts[1].uv = glm::vec2(0.0f,    tile_v);
    verts[2].uv = glm::vec2(tile_u,  tile_v);
    verts[3].uv = glm::vec2(tile_u,  0.0f);

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    glBindVertexArray(0);

    out.idx_count = (unsigned int)idxs.size();
}

// трава обочина
void make_grass_mesh(mesh& out) {
    std::vector<vertex> verts(4);
    std::vector<unsigned int> idxs = {0, 1, 2, 0, 2, 3};

    float half_len = track_half_len;
    float grass_w  = 15.0f;
    float y        = -0.01f;   // чуть ниже дороги

    verts[0].pos = glm::vec3(track_half_w,           y, -half_len);
    verts[1].pos = glm::vec3(track_half_w,           y,  half_len);
    verts[2].pos = glm::vec3(track_half_w + grass_w, y,  half_len);
    verts[3].pos = glm::vec3(track_half_w + grass_w, y, -half_len);

    for (int i = 0; i < 4; ++i)
        verts[i].nrm = glm::vec3(0, 1, 0);

    float tile_u = 3.0f;
    float tile_v = 80.0f;

    verts[0].uv = glm::vec2(0.0f,    0.0f);
    verts[1].uv = glm::vec2(0.0f,    tile_v);
    verts[2].uv = glm::vec2(tile_u,  tile_v);
    verts[3].uv = glm::vec2(tile_u,  0.0f);

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    glBindVertexArray(0);

    out.idx_count = (unsigned int)idxs.size();
}

// фонарь обочина
void make_lamp_mesh(mesh& out) {
    std::vector<vertex> verts;
    std::vector<unsigned int> idxs;

    float pole_half_w = 0.15f;
    float pole_h      = lamp_y - 0.5f;

    glm::vec3 pole_min(-pole_half_w, 0.0f, -pole_half_w);
    glm::vec3 pole_max( pole_half_w, pole_h,  pole_half_w);
    add_box(verts, idxs, pole_min, pole_max);

    float head_h = 0.3f;
    glm::vec3 head_min(-0.4f, pole_h, -0.4f);
    glm::vec3 head_max( 0.4f, pole_h + head_h, 0.4f);
    add_box(verts, idxs, head_min, head_max);

    glm::vec3 sph_center(0.0f, lamp_y, 0.0f);
    float sph_r = 0.5f;
    add_sphere(verts, idxs, sph_center, sph_r, 10, 16);

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    glBindVertexArray(0);

    out.idx_count = (unsigned int)idxs.size();

    std::cout << "[env] lamp mesh verts: " << verts.size()
              << " idx: " << idxs.size() << std::endl;
}

// сфера свет
void make_light_sphere_mesh(mesh& out) {
    std::vector<vertex> verts;
    std::vector<unsigned int> idxs;

    add_sphere(verts, idxs, glm::vec3(0.0f), 1.0f, 8, 12);

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    glBindVertexArray(0);

    out.idx_count = (unsigned int)idxs.size();
}

// дерево
void make_tree_mesh(mesh& trunk, mesh& leaves) {
    std::vector<vertex> v_trunk;
    std::vector<unsigned int> i_trunk;

    std::vector<vertex> v_leaves;
    std::vector<unsigned int> i_leaves;

    // ствол
    add_box(v_trunk, i_trunk,
            glm::vec3(-0.2f, 0.0f, -0.2f),
            glm::vec3( 0.2f, 2.0f,  0.2f));

    // листики
    add_sphere(v_leaves, i_leaves,
               glm::vec3(0.0f, 2.6f, 0.0f),
               1.0f, 8, 12);

    // ствол
    glGenVertexArrays(1, &trunk.vao);
    glGenBuffers(1, &trunk.vbo);
    glGenBuffers(1, &trunk.ebo);

    glBindVertexArray(trunk.vao);
    glBindBuffer(GL_ARRAY_BUFFER, trunk.vbo);
    glBufferData(GL_ARRAY_BUFFER, v_trunk.size() * sizeof(vertex), v_trunk.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, trunk.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_trunk.size() * sizeof(unsigned int), i_trunk.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    trunk.idx_count = (unsigned int)i_trunk.size();

    // листики
    glGenVertexArrays(1, &leaves.vao);
    glGenBuffers(1, &leaves.vbo);
    glGenBuffers(1, &leaves.ebo);

    glBindVertexArray(leaves.vao);
    glBindBuffer(GL_ARRAY_BUFFER, leaves.vbo);
    glBufferData(GL_ARRAY_BUFFER, v_leaves.size() * sizeof(vertex), v_leaves.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, leaves.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_leaves.size() * sizeof(unsigned int), i_leaves.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    leaves.idx_count = (unsigned int)i_leaves.size();

    glBindVertexArray(0);
}

// скамеечка
void make_bench_mesh(mesh& wood, mesh& metal) {
    std::vector<vertex> v_wood;
    std::vector<unsigned int> i_wood;

    std::vector<vertex> v_metal;
    std::vector<unsigned int> i_metal;

    // сиденье
    add_box(v_wood, i_wood,
            glm::vec3(-1.0f, 0.5f, -0.3f),
            glm::vec3( 1.0f, 0.7f,  0.3f));
    // спинка
    add_box(v_wood, i_wood,
            glm::vec3(-1.0f, 0.7f, -0.1f),
            glm::vec3( 1.0f, 1.1f,  0.1f));

    // ножки
    add_box(v_metal, i_metal,
            glm::vec3(-0.9f, 0.0f, -0.2f),
            glm::vec3(-0.7f, 0.5f,  0.2f));
    add_box(v_metal, i_metal,
            glm::vec3( 0.7f, 0.0f, -0.2f),
            glm::vec3( 0.9f, 0.5f,  0.2f));

    // дерево
    glGenVertexArrays(1, &wood.vao);
    glGenBuffers(1, &wood.vbo);
    glGenBuffers(1, &wood.ebo);

    glBindVertexArray(wood.vao);
    glBindBuffer(GL_ARRAY_BUFFER, wood.vbo);
    glBufferData(GL_ARRAY_BUFFER, v_wood.size() * sizeof(vertex), v_wood.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wood.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_wood.size() * sizeof(unsigned int), i_wood.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    wood.idx_count = (unsigned int)i_wood.size();

    // железка
    glGenVertexArrays(1, &metal.vao);
    glGenBuffers(1, &metal.vbo);
    glGenBuffers(1, &metal.ebo);

    glBindVertexArray(metal.vao);
    glBindBuffer(GL_ARRAY_BUFFER, metal.vbo);
    glBufferData(GL_ARRAY_BUFFER, v_metal.size() * sizeof(vertex), v_metal.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, metal.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_metal.size() * sizeof(unsigned int), i_metal.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    metal.idx_count = (unsigned int)i_metal.size();

    glBindVertexArray(0);
}

// кустик
void make_bush_mesh(mesh& m) {
    std::vector<vertex> verts;
    std::vector<unsigned int> idxs;

    add_sphere(verts, idxs, glm::vec3(0.0f, 0.7f, 0.0f), 0.8f, 8, 12);

    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    m.idx_count = (unsigned int)idxs.size();

    glBindVertexArray(0);
}

// отбойник
void make_rail_mesh(mesh& m) {
    std::vector<vertex> verts;
    std::vector<unsigned int> idxs;

    // один сегмент отбойника
    float seg = rail_seg_len * 0.5f;
    add_box(verts, idxs,
            glm::vec3(-0.15f, 0.5f, -seg),
            glm::vec3( 0.15f, 1.1f,  seg));

    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    m.idx_count = (unsigned int)idxs.size();

    glBindVertexArray(0);
}

// машины трафика
void make_traffic_car_mesh(mesh& m) {
    std::vector<vertex> verts;
    std::vector<unsigned int> idxs;

    // низ
    add_box(verts, idxs,
            glm::vec3(-0.6f, 0.0f, -1.2f),
            glm::vec3( 0.6f, 0.4f,  1.2f));

    // верх
    add_box(verts, idxs,
            glm::vec3(-0.4f, 0.4f, -0.3f),
            glm::vec3( 0.4f, 0.9f,  0.6f));

    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(vertex), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.size() * sizeof(unsigned int), idxs.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, uv));

    m.idx_count = (unsigned int)idxs.size();

    glBindVertexArray(0);

    std::cout << "[traffic] car mesh verts: " << verts.size()
              << " idx: " << idxs.size() << std::endl;
}

// init  фонарей
void init_lamps() {
    lamp_pos.clear();

    float min_z = -track_half_len + 20.0f;
    float max_z =  track_half_len - 20.0f;
    float offset_x = track_half_w + 4.0f;

    for (float z = min_z; z <= max_z; z += lamp_step) {
        lamp_pos.push_back(glm::vec3(-offset_x, lamp_y, z));
        lamp_pos.push_back(glm::vec3( offset_x, lamp_y, z));
    }

    std::cout << "[env] lamps: " << lamp_pos.size() << std::endl;
}

// init окружения
void init_env() {
    tree_pos.clear();
    bench_pos.clear();
    bush_pos.clear();

    float min_z = -track_half_len + 60.0f;
    float max_z =  track_half_len - 60.0f;
    float tree_off  = track_half_w + 10.0f;
    float bush_off  = track_half_w + 5.0f;

    // деревья скамейки вдоль трассы
    for (float z = min_z; z <= max_z; z += 80.0f) {
        glm::vec3 left_tree  = glm::vec3(-tree_off, 0.0f, z + 15.0f);
        glm::vec3 right_tree = glm::vec3( tree_off, 0.0f, z - 15.0f);

        tree_pos.push_back(left_tree);
        tree_pos.push_back(right_tree);

        glm::vec3 left_bench  = left_tree;
        glm::vec3 right_bench = right_tree;

        left_bench.x  += 2.0f;
        right_bench.x -= 2.0f;

        bench_pos.push_back(left_bench);
        bench_pos.push_back(right_bench);
    }

    // кусты вдоль трассы
    for (float z = min_z; z <= max_z; z += 80.0f) {
        bush_pos.push_back(glm::vec3(-bush_off, 0.0f, z + 10.0f));
        bush_pos.push_back(glm::vec3( bush_off, 0.0f, z - 10.0f));
    }

    std::cout << "[env] trees: " << tree_pos.size()
              << " benches: " << bench_pos.size()
              << " bushes: " << bush_pos.size() << std::endl;
}

const float TRAFFIC_COLLISION_DIST = 0.6f;
const float TRAFFIC_SAFE_DIST     = 15.0f;
const float TRAFFIC_PUSH_FACTOR   = 0.0f;
const float TRAFFIC_MAX_PUSH_PER_FRAME = 0.0f;

// хитбокс игрока и трафика
const float PLAYER_HALF_W = 0.95f;
const float PLAYER_HALF_L = 2.5f;

const float TRAFFIC_HALF_W = 0.85f;
const float TRAFFIC_HALF_L = 2.2f;


// init трафика
void init_traffic() {
    traffic.clear();
    float lane_base = track_half_w * 0.25f;
    std::vector<float> lane_x = {
        lane_base * 2.4f,
        lane_base * 0.8f,
        -lane_base * 0.8f,
        -lane_base * 2.4f
    };

    float base_y = 0.5f;
    float forward_start  = -track_half_len + 200.0f;
    float backward_start =  track_half_len - 200.0f;
    int cars_per_lane = 400;
    float spacing = 80.0f;

    // патилтра трафика
    glm::vec3 colors[] = {
        glm::vec3(1.0f, 0.1f, 0.1f),
        glm::vec3(0.1f, 0.4f, 1.0f),
        glm::vec3(0.1f, 0.8f, 0.2f),
        glm::vec3(1.0f, 0.8f, 0.1f),
        glm::vec3(0.9f, 0.3f, 0.8f),
        glm::vec3(0.8f, 0.8f, 0.8f)
    };
    int color_count = (int)(sizeof(colors) / sizeof(colors[0]));

    for (size_t li = 0; li < lane_x.size(); ++li) {
        float lx = lane_x[li];
        float dir = (lx > 0.0f) ? 1.0f : -1.0f;

        for (int i = 0; i < cars_per_lane; ++i) {
            traffic_car tc;
            tc.pos.y = base_y;
            tc.pos.x = lx;
            if (dir > 0.0f) tc.pos.z = forward_start + (float)i * spacing;
            else            tc.pos.z = backward_start - (float)i * spacing;

            tc.dir = dir;
            float jitter_z = ((float)rand() / RAND_MAX - 0.5f) * spacing * 0.25f;
            if (dir > 0.0f) tc.pos.z = forward_start + (float)i * spacing + jitter_z;
            else            tc.pos.z = backward_start - (float)i * spacing + jitter_z;

            // случайная скорость
            float base_speed = 7.0f + ((float)rand() / RAND_MAX) * 1.2f;
            tc.speed = base_speed * (0.9f + ((float)rand() / RAND_MAX) * 0.2f);
            tc.max_speed = base_speed * 1.4f;

            tc.color = colors[i % color_count];
            tc.stopped = false;
            tc.stop_timer = 0.0f;
            tc.mass = 1.0f + ((float)rand() / RAND_MAX) * 0.8f;
            traffic.push_back(tc);
        }
    }

    std::cout << "[traffic] cars: " << traffic.size() << " (4 lanes: " << lane_x.size() << ")" << std::endl;
}
// логика трафика
void upd_traffic(float dt) {
    if (traffic.empty()) return;

    const float lane_threshold = 1.0f;
    const float safe_dist = TRAFFIC_SAFE_DIST;
    const float max_brake = 8.0f;
    const float speed_accel_rate = 2.5f;

    const float collision_speed_thresh = 1.0f;
    const float stop_duration = 5.0f;

    const float traffic_collision_dist = TRAFFIC_COLLISION_DIST;
    const float push_factor = TRAFFIC_PUSH_FACTOR;
    const float max_push_per_frame = TRAFFIC_MAX_PUSH_PER_FRAME;
    // добавление трафика
    for (size_t i = 0; i < traffic.size(); ++i) {
        auto &tc = traffic[i];
        glm::vec3 vec_to_tc = tc.pos - car.pos;
        float dist2 = glm::dot(vec_to_tc, vec_to_tc);
        if (dist2 < 1e-6f) continue;
        glm::vec3 dir_to_tc = glm::normalize(vec_to_tc);

        float rel_speed_norm = glm::dot(car.vel - glm::vec3(0.0f,0.0f,0.0f), dir_to_tc);

        float dx = fabs(tc.pos.x - car.pos.x);
        float dz = fabs(tc.pos.z - car.pos.z);

        bool hit =
            dx < (PLAYER_HALF_W + TRAFFIC_HALF_W) &&
            dz < (PLAYER_HALF_L + TRAFFIC_HALF_L);

        const float COLLIDE_NORM_SPEED_THRESH = 0.6f;

        // коллизия с игроком
        if (!tc.stopped && hit && rel_speed_norm > COLLIDE_NORM_SPEED_THRESH) {
            tc.stopped = true;
            tc.stop_timer = stop_duration;
            tc.speed = 0.0f;
            car.pos = prev_car_pos;
            car.vel = glm::vec3(0.0f, 0.0f, 0.0f);
            car.brake = true;
        }
    }

    std::vector<std::vector<int>> lanes;
    std::vector<float> lane_x_values;

    // по полосам
    for (size_t i = 0; i < traffic.size(); ++i) {
        float x = traffic[i].pos.x;
        bool placed = false;
        for (size_t li = 0; li < lane_x_values.size(); ++li) {
            if (fabs(x - lane_x_values[li]) < lane_threshold) {
                lanes[li].push_back((int)i);
                lane_x_values[li] = (lane_x_values[li] * 0.7f + x * 0.3f);
                placed = true;
                break;
            }
        }
        if (!placed) {
            lane_x_values.push_back(x);
            lanes.push_back(std::vector<int>(1, (int)i));
        }
    }
    // обработка каждой полосы
    for (size_t li = 0; li < lanes.size(); ++li) {
        auto &idxs = lanes[li];
        if (idxs.empty()) continue;

        float dir = traffic[idxs[0]].dir;

        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            if (dir > 0) return traffic[a].pos.z < traffic[b].pos.z;
            else return traffic[a].pos.z > traffic[b].pos.z;
        });

        for (int k = (int)idxs.size() - 1; k >= 0; --k) {
            int i = idxs[k];
            auto &a = traffic[i];

            if (a.stopped) {
                a.stop_timer -= dt;
                if (a.stop_timer <= 0.0f) {
                    a.stopped = false;
                    a.stop_timer = 0.0f;
                    a.speed = glm::min(a.max_speed * 0.6f, a.max_speed);
                } else {
                    a.speed = 0.0f;
                    continue;
                }
            }

            float vmax = (a.max_speed > 0.0f) ? a.max_speed : 4.0f;
            float target_speed = vmax;

            // дистанция
            if (k < (int)idxs.size() - 1) {
                int idx_ahead = idxs[k + 1];
                auto &ahead = traffic[idx_ahead];
                float gap = (ahead.pos.z - a.pos.z) * a.dir;
                float rel_speed = a.speed - ahead.speed;
                float dynamic_safe = safe_dist;
                if (ahead.stopped || ahead.speed < TRAFFIC_SLOW_SPEED_THRESH) {
                    dynamic_safe = glm::max(dynamic_safe, TRAFFIC_STOP_BUFFER_DIST);
                }

                if (gap < dynamic_safe) {
                    if (ahead.stopped || ahead.speed < TRAFFIC_SLOW_SPEED_THRESH) {
                        float frac = glm::clamp((gap / dynamic_safe), 0.0f, 1.0f);
                        target_speed = glm::min(target_speed, ahead.speed * (0.3f + 0.7f * frac));
                        if (gap < 0.8f) {
                            target_speed = 0.0f;
                        }
                        if (rel_speed > 1.5f && gap < (dynamic_safe * 0.5f)) {
                            a.stopped = true;
                            a.stop_timer = 0.6f;
                            a.speed = 0.0f;
                        }
                    } else {
                        float ratio = glm::clamp((gap / dynamic_safe), 0.0f, 1.0f);
                        float ahead_speed = ahead.speed;
                        target_speed = glm::min(target_speed, ahead_speed * (0.95f + 0.05f * ratio));
                        if (gap < dynamic_safe * 0.6f) {
                            target_speed = glm::min(target_speed, ahead_speed * 0.6f);
                        }
                    }
                }
                if (push_factor > 0.001f && gap < traffic_collision_dist && rel_speed > collision_speed_thresh) {
                    float push = push_factor * a.speed * a.mass;
                    float push_z = glm::clamp(push * dt, -max_push_per_frame, max_push_per_frame);
                    ahead.pos.z += push_z * ahead.dir;
                    a.speed *= 0.25f;
                    if (push * dt > 0.05f) {
                        ahead.stopped = true;
                        ahead.stop_timer = glm::min(stop_duration * 0.5f, 2.0f);
                        ahead.speed = 0.0f;
                    }
                }
            }
            float dv = target_speed - a.speed;
            float accel = 0.0f;
            if (dv > 0.0f) accel = glm::min(dv / dt, speed_accel_rate);
            else          accel = glm::max(dv / dt, -max_brake);

            a.speed += accel * dt;
            if (a.speed < 0.0f) a.speed = 0.0f;
            if (a.speed > vmax) a.speed = vmax;

            a.pos.z += a.speed * dt * a.dir;

            if (k < (int)idxs.size() - 1) {
                int idx_ahead = idxs[k + 1];
                auto &ahead = traffic[idx_ahead];
                float gap = (ahead.pos.z - a.pos.z) * a.dir;
                if (gap < 0.3f) {
                    a.pos.z = ahead.pos.z - a.dir * 0.3f;
                    a.speed = glm::min(a.speed, ahead.speed);
                }
            }
        }
    }
}

// обработка клавиартуры
void key_cb(GLFWwindow* w, int key, int scancode, int action, int mods) {
    (void)w; (void)scancode; (void)mods;

    bool down = (action == GLFW_PRESS || action == GLFW_REPEAT);
    if (key == GLFW_KEY_W) key_w = down;
    if (key == GLFW_KEY_S) key_s = down;
    if (key == GLFW_KEY_A) key_a = down;
    if (key == GLFW_KEY_D) key_d = down;
    if (key == GLFW_KEY_SPACE) key_space = down;
    if (key == GLFW_KEY_R && action == GLFW_PRESS) key_r = true;
    if (key == GLFW_KEY_J && action == GLFW_PRESS) {
        gate_open = !gate_open;
    }
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
        orbit_mode = false;
        orbit_az = 0.0f;
        orbit_el = glm::radians(20.0f);
        orbit_dist = cam_dist;
        cam.pos = car.pos + glm::vec3(0.0f, cam_height, -cam_dist);
        cam.target = car.pos + glm::vec3(0.0f, cam_target_h, 0.0f);
    }
    if (key == GLFW_KEY_KP_ADD && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        orbit_dist = glm::clamp(orbit_dist * orbit_zoom_step, orbit_min_dist, orbit_max_dist);
    }
    if (key == GLFW_KEY_KP_SUBTRACT && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        orbit_dist = glm::clamp(orbit_dist / orbit_zoom_step, orbit_min_dist, orbit_max_dist);
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) key_esc = true;
}

// мышь кнопки
void mouse_button_cb(GLFWwindow* w, int button, int action, int mods) {
    (void)w; (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouse_down = true;
            double x, y;
            glfwGetCursorPos(win, &x, &y);
            last_mouse_x = x;
            last_mouse_y = y;
            orbit_mode = true;
        } else if (action == GLFW_RELEASE) {
            mouse_down = false;
        }
    }
}

// мышь движение
void cursor_pos_cb(GLFWwindow* w, double xpos, double ypos) {
    (void)w;
    if (!mouse_down) return;
    double dx = xpos - last_mouse_x;
    double dy = ypos - last_mouse_y;
    last_mouse_x = xpos;
    last_mouse_y = ypos;

    orbit_az += (float)dx * mouse_sens;
    orbit_el += (float)dy * mouse_sens;

    if (orbit_el > orbit_max_el) orbit_el = orbit_max_el;
    if (orbit_el < orbit_min_el) orbit_el = orbit_min_el;
}

// зум камеры
void scroll_cb(GLFWwindow* w, double xoffset, double yoffset) {
    (void)w; (void)xoffset;
    if (yoffset > 0) {
        orbit_dist *= orbit_zoom_step;\
    } else if (yoffset < 0) {
        orbit_dist /= orbit_zoom_step;
    }
    orbit_dist = glm::clamp(orbit_dist, orbit_min_dist, orbit_max_dist);
}

// сброс позиции
void reset_car() {
    car.pos = glm::vec3(0.0f, 0.5f, -900.0f);
    car.vel = glm::vec3(0.0f);
    car.yaw = 0.0f;
    car.steer = 0.0f;
    car.brake = false;
    gate_z = car.pos.z + 3.0f;
    gate_open = false;
}

void upd_car(float dt) {
    float acc_fwd = 30.0f;
    float acc_back = 20.0f;
    float max_speed = 80.0f;
    float friction = 4.0f;
    float steer_speed  = 3.5f;
    float steer_return = 5.0f;
    float steer_max    = 1.0f;
    float throttle = 0.0f;

    // управление машиной
    if (key_w) throttle += 1.0f;
    if (key_s) throttle -= 1.0f;

    if (key_a) car.steer += steer_speed * dt;
    if (key_d) car.steer -= steer_speed * dt;

    if (!key_a && !key_d) {
        if (car.steer > 0.0f) {
            car.steer -= steer_return * dt;
            if (car.steer < 0.0f) car.steer = 0.0f;
        } else if (car.steer < 0.0f) {
            car.steer += steer_return * dt;
            if (car.steer > 0.0f) car.steer = 0.0f;
        }
    }

    if (car.steer > steer_max) car.steer = steer_max;
    if (car.steer < -steer_max) car.steer = -steer_max;

    glm::vec3 fwd = glm::vec3(std::sin(car.yaw), 0.0f, std::cos(car.yaw));
    float speed = glm::dot(car.vel, fwd);

    float acc = 0.0f;
    if (throttle > 0.0f) acc = acc_fwd * throttle;
    else if (throttle < 0.0f) acc = acc_back * throttle;

    speed += acc * dt;

    // пробел тормоз
    bool braking = key_space;
    if (braking) {
        float brake_force = 70.0f;
        if (speed > 0.0f) {
            speed -= brake_force * dt;
            if (speed < 0.0f) speed = 0.0f;
        } else if (speed < 0.0f) {
            speed += brake_force * dt;
            if (speed > 0.0f) speed = 0.0f;
        }
    }

    if (std::abs(speed) > 0.01f) {
        float sign = (speed > 0.0f) ? 1.0f : -1.0f;
        speed -= sign * friction * dt;
        if (sign > 0.0f && speed < 0.0f) speed = 0.0f;
        if (sign < 0.0f && speed > 0.0f) speed = 0.0f;
    } else {
        speed = 0.0f;
    }

    if (speed > max_speed) speed = max_speed;
    if (speed < -max_speed * 0.3f) speed = -max_speed * 0.3f;

    float rel_speed = std::abs(speed) / max_speed;
    float speed_norm = glm::clamp((rel_speed - 0.1f) / 0.5f, 0.0f, 1.0f);
    speed_norm = 0.3f + 0.7f * speed_norm;

    float turn_factor = 1.2f;
    float yaw_delta = car.steer * speed_norm * turn_factor;
    float dir_sign = (speed >= 0.0f ? 1.0f : -1.0f);
    car.yaw += yaw_delta * dt * dir_sign;

    car.vel = fwd * speed;
    car.pos += car.vel * dt;
    car.pos.y = 0.5f;

    float wrap = track_half_len - 5.0f;
    if (car.pos.z > wrap) {
        car.pos.z -= 2.0f * wrap;
    } else if (car.pos.z < -wrap) {
        car.pos.z += 2.0f * wrap;
    }

    // коллизия с отбойниками
    float wall_x = track_half_w * guard_x_factor;
    if (car.pos.x > wall_x) {
        car.pos.x = wall_x;
        car.vel.x *= -0.3f;
    } else if (car.pos.x < -wall_x) {
        car.pos.x = -wall_x;
        car.vel.x *= -0.3f;
    }
    // коллизия шлагбаум
    if (!gate_open) {
        bool hit_x = fabs(car.pos.x - gate_x) < gate_solid_half;

        bool approaching_forward  = (car.vel.z > 0.0f && car.pos.z < gate_z);
        bool approaching_backward = (car.vel.z < 0.0f && car.pos.z > gate_z);

        if (hit_x && (approaching_forward || approaching_backward)) {
            float next_z = car.pos.z + car.vel.z * dt;
            if (approaching_forward && next_z >= gate_z) {
                car.pos.z = gate_z - 0.05f;
                car.vel.z *= -0.2f;
            } else if (approaching_backward && next_z <= gate_z) {
                car.pos.z = gate_z + 0.05f;
                car.vel.z *= -0.2f;
            }
        }
    }

    car.brake = braking && std::abs(speed) > 1.0f;
}

// обновление камеры
void upd_cam(float dt) {
    glm::vec3 target = car.pos + glm::vec3(0.0f, cam_target_h, 0.0f);
    // свободный обзор
    if (orbit_mode) {
        float ca = std::cos(orbit_az), sa = std::sin(orbit_az);
        float ce = std::cos(orbit_el), se = std::sin(orbit_el);

        glm::vec3 offset;
        offset.x = orbit_dist * ce * sa;
        offset.y = orbit_dist * se;
        offset.z = orbit_dist * ce * ca;

        glm::vec3 desired_pos = target + offset;

        float world_min_y = min_cam_height;
        if (desired_pos.y < world_min_y) {
            desired_pos.y = world_min_y;
            float clamped_sin = glm::clamp(desired_pos.y - target.y, -orbit_dist, orbit_dist) / orbit_dist;
            clamped_sin = glm::clamp(clamped_sin, -0.9999f, 0.9999f);
            orbit_el = asinf(clamped_sin);
            orbit_el = glm::clamp(orbit_el, orbit_min_el, orbit_max_el);
        }

        if (dt > 0.0f) {
            float t = 1.0f - std::exp(-cam_spring * dt);
            cam.pos = glm::mix(cam.pos, desired_pos, t);
        } else {
            cam.pos = desired_pos;
        }

        cam.target = target;
        cam.up = glm::vec3(0.0f, 1.0f, 0.0f);
    // фиксированная камера
    } else {
        glm::vec3 fwd = glm::vec3(std::sin(car.yaw), 0.0f, std::cos(car.yaw));
        glm::vec3 upv = glm::vec3(0.0f, 1.0f, 0.0f);

        glm::vec3 desired_pos = car.pos - fwd * cam_dist + upv * cam_height;
        glm::vec3 target_loc = car.pos + glm::vec3(0.0f, cam_target_h, 0.0f);

        if (desired_pos.y < min_cam_height) desired_pos.y = min_cam_height;

        if (dt > 0.0f) {
            float t = 1.0f - std::exp(-cam_spring * dt);
            cam.pos = glm::mix(cam.pos, desired_pos, t);
        } else {
            cam.pos = desired_pos;
        }

        cam.target = target_loc;
        cam.up = upv;
    }
}

// рисование меша
void draw_mesh(const mesh& m) {
    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.idx_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// рисование света сферы
void draw_light_sphere(const glm::mat4& vp,
                       const glm::vec3& pos,
                       float radius,
                       const glm::vec3& emis_col,
                       GLint loc_m, GLint loc_emis, GLint loc_mat_col)
{
    (void)vp;
    glm::mat4 m = glm::mat4(1.0f);
    m = glm::translate(m, pos);
    m = glm::scale(m, glm::vec3(radius));

    glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m));
    glUniform3f(loc_mat_col, 0.0f, 0.0f, 0.0f);
    glUniform3fv(loc_emis, 1, glm::value_ptr(emis_col));

    draw_mesh(light_sphere_mesh);
}

// init
bool init_gl() {
    if (!glfwInit()) {
        std::cerr << "[err] glfw init\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    win = glfwCreateWindow(win_w, win_h, "night road", nullptr, nullptr);
    if (!win) {
        std::cerr << "[err] window create\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(win, key_cb);

    glfwSetMouseButtonCallback(win, mouse_button_cb);
    glfwSetCursorPosCallback(win, cursor_pos_cb);
    glfwSetScrollCallback(win, scroll_cb);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[err] glad init\n";
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    std::cout << "[info] opengl version: " << glGetString(GL_VERSION) << std::endl;
    return true;
}

void make_gate_mesh(mesh &m);
void draw_gate();
extern mesh gate_mesh;

// загрузка контента
bool load_content() {
    prog = make_prog(vs_src, fs_src);
    if (!prog) {
        std::cerr << "[err] no shader\n";
        return false;
    }

    std::cout << "[info] loading car.obj\n";
    if (!load_obj_with_mtl("car.obj", car_model)) {
        std::cout << "[info] load obj single mesh\n";
        if (!load_obj("car.obj", car_model.m)) return false;
    }

    std::cout << "[info] creating track\n";
    make_track_mesh(track_model.m);

    std::cout << "[info] creating lamps\n";
    make_lamp_mesh(lamp_mesh);
    init_lamps();

    make_gate_mesh(gate_mesh);

    std::cout << "[info] loading track_diffuse.png\n";
    unsigned int track_tex_file = load_tex("track_diffuse.png");
    unsigned int track_tex_proc = make_road_texture();
    track_model.tex = track_tex_proc ? track_tex_proc : track_tex_file;
    if (!track_model.tex) return false;

    std::cout << "[info] creating env\n";
    make_light_sphere_mesh(light_sphere_mesh);
    make_tree_mesh(tree_trunk_model.m, tree_leaves_model.m);
    make_bench_mesh(bench_wood_model.m, bench_metal_model.m);
    make_bush_mesh(bush_model.m);
    make_rail_mesh(rail_model.m);
    make_grass_mesh(grass_model.m);
    make_traffic_car_mesh(traffic_car_model.m);
    init_env();
    init_traffic();

    std::cout << "[info] loading textures\n";
    tree_trunk_model.tex   = load_tex("tree_bark.png");
    tree_leaves_model.tex  = load_tex("tree_leaves.png");
    bench_wood_model.tex   = load_tex("bench_wood.png");
    bench_metal_model.tex  = load_tex("bench_metal.png");
    bush_model.tex         = load_tex("bush.png");
    rail_model.tex         = load_tex("rail.png");
    grass_model.tex        = load_tex("grass.png");

    traffic_car_model.tex  = make_traffic_car_texture();

    glUseProgram(prog);
    int loc_tex = glGetUniformLocation(prog, "u_tex");
    glUniform1i(loc_tex, 0);

    reset_car();
    prev_car_pos = car.pos;
    cam.pos = car.pos + glm::vec3(0.0f, cam_height, -cam_dist);
    cam.target = car.pos + glm::vec3(0.0f, cam_target_h, 0.0f);
    cam.up = glm::vec3(0,1,0);

    std::cout << "[info] content loaded\n";
    return true;
}

// меш шлагбаум
void make_gate_mesh(mesh &m) {
    float verts[] = {
        //  x     y     z  норм x y z             u     v
        -0.5f, 0.0f, 0.0f,      0,1,0,         0.0f, 0.0f,
         0.5f, 0.0f, 0.0f,      0,1,0,         1.0f, 0.0f,
         0.5f, 1.0f, 0.0f,      0,1,0,         1.0f, 1.0f,
        -0.5f, 1.0f, 0.0f,      0,1,0,         0.0f, 1.0f
    };
    unsigned int idxs[] = { 0,1,2,  2,3,0 };

    GLuint vao=0, vbo=0, ebo=0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idxs), idxs, GL_STATIC_DRAW);

    // assuming shader layout: 0 = position(vec3), 1 = normal(vec3), 2 = uv(vec2)
    GLsizei stride = (3+3+2) * sizeof(float);
    // pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)(0));
    // normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    // uv
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m.vao = vao;
    m.idx_count = 6;
    m.vbo = vbo;
    m.ebo = ebo;
}

// шлагбаум
void draw_gate() {
    glUseProgram(prog);

    float height = 0.12f;
    float length = gate_half_width * 2;
    float hinge_y = 0.6f;

    glm::vec3 hinge_world = glm::vec3(gate_x - length * 0.5f, hinge_y, gate_z);

    glm::mat4 m(1.0f);
    m = glm::translate(m, hinge_world);

    m = glm::rotate(m, gate_angle, glm::vec3(0.0f, 0.0f, 1.0f));

    m = glm::translate(m, glm::vec3(length * 0.5f, 0.0f, 0.0f));
    m = glm::scale(m, glm::vec3(length, height, 1.0f));

    int loc_m = glGetUniformLocation(prog, "u_m");
    int loc_emis = glGetUniformLocation(prog, "u_emis");
    int loc_mat_col = glGetUniformLocation(prog, "u_mat_col");

    if (loc_m >= 0) glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m));
    if (loc_emis >= 0) glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);

    if (gate_open) {
        if (loc_mat_col >= 0) glUniform3f(loc_mat_col, 0.2f, 0.8f, 0.2f);
    } else {
        if (loc_mat_col >= 0) glUniform3f(loc_mat_col, 0.9f, 0.2f, 0.2f);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, track_model.tex);

    draw_mesh(gate_mesh);
}

// отрисовка сцены
void draw_scene() {
    glClearColor(0.01f, 0.02f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(prog);

    glm::mat4 proj = glm::perspective(glm::radians(60.0f), (float)win_w / (float)win_h, 0.1f, 5000.0f);
    glm::mat4 v = glm::lookAt(cam.pos, cam.target, cam.up);

    int loc_m   = glGetUniformLocation(prog, "u_m");
    int loc_v   = glGetUniformLocation(prog, "u_v");
    int loc_p   = glGetUniformLocation(prog, "u_p");
    int loc_light = glGetUniformLocation(prog, "u_light_dir");
    int loc_cam = glGetUniformLocation(prog, "u_cam_pos");
    int loc_amb = glGetUniformLocation(prog, "u_amb");
    int loc_dir_int = glGetUniformLocation(prog, "u_dir_int");
    int loc_emis = glGetUniformLocation(prog, "u_emis");
    int loc_mat_col = glGetUniformLocation(prog, "u_mat_col");

    int loc_pl_count = glGetUniformLocation(prog, "u_pl_count");
    int loc_pl_pos   = glGetUniformLocation(prog, "u_pl_pos");
    int loc_pl_col   = glGetUniformLocation(prog, "u_pl_col");
    int loc_pl_rad   = glGetUniformLocation(prog, "u_pl_rad");
    int loc_pl_dir   = glGetUniformLocation(prog, "u_pl_dir");
    int loc_pl_cos   = glGetUniformLocation(prog, "u_pl_cos");

    glUniformMatrix4fv(loc_v, 1, GL_FALSE, glm::value_ptr(v));
    glUniformMatrix4fv(loc_p, 1, GL_FALSE, glm::value_ptr(proj));
    glUniform3fv(loc_light, 1, glm::value_ptr(dir_dir));
    glUniform3fv(loc_cam, 1, glm::value_ptr(cam.pos));

    glUniform3fv(loc_amb, 1, glm::value_ptr(amb_col));
    glUniform1f(loc_dir_int, dir_int);

    glm::vec3 pl_pos[max_pl];
    glm::vec3 pl_col[max_pl];
    glm::vec3 pl_dir[max_pl];
    float     pl_rad[max_pl];
    float     pl_cos[max_pl];
    int pl_count = 0;

    glm::vec3 fwd = glm::vec3(std::sin(car.yaw), 0.0f, std::cos(car.yaw));
    glm::vec3 right = glm::vec3(fwd.z, 0.0f, -fwd.x);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    // ближний свет
    {
        glm::vec3 head_base = car.pos + up * hl_y + fwd * hl_fwd;
        float hl_cos = std::cos(glm::radians(hl_ang_deg));
        glm::vec3 hl_dir_vec = glm::normalize(glm::vec3(fwd.x, -0.18f, fwd.z));

        if (pl_count < max_pl) {
            pl_pos[pl_count] = head_base + right * hl_side;
            pl_col[pl_count] = hl_col;
            pl_rad[pl_count] = hl_rad;
            pl_dir[pl_count] = hl_dir_vec;
            pl_cos[pl_count] = hl_cos;
            ++pl_count;
        }
        if (pl_count < max_pl) {
            pl_pos[pl_count] = head_base - right * hl_side;
            pl_col[pl_count] = hl_col;
            pl_rad[pl_count] = hl_rad;
            pl_dir[pl_count] = hl_dir_vec;
            pl_cos[pl_count] = hl_cos;
            ++pl_count;
        }
    }

    // стопы и габариты

    {
        glm::vec3 tail_base = car.pos + up * tail_y - fwd * tail_back;

        glm::vec3 tail_col = tail_col_idle;
        float tail_rad = tail_rad_idle;

        if (car.brake) {
            tail_col = tail_col_brake;
            tail_rad = tail_rad_brake;
        }

        float tcos = std::cos(glm::radians(tail_ang_deg));
        glm::vec3 tdir = glm::normalize(glm::vec3(-fwd.x, -0.05f, -fwd.z));

        if (pl_count < max_pl) {
            pl_pos[pl_count] = tail_base + right * tail_side;
            pl_col[pl_count] = tail_col;
            pl_rad[pl_count] = tail_rad;
            pl_dir[pl_count] = tdir;
            pl_cos[pl_count] = tcos;
            ++pl_count;
        }
        if (pl_count < max_pl) {
            pl_pos[pl_count] = tail_base - right * tail_side;
            pl_col[pl_count] = tail_col;
            pl_rad[pl_count] = tail_rad;
            pl_dir[pl_count] = tdir;
            pl_cos[pl_count] = tcos;
            ++pl_count;
        }
    }

    // свет фонарией
    for (size_t i = 0; i < lamp_pos.size() && pl_count < max_pl; ++i) {
        if (std::abs(lamp_pos[i].z - car.pos.z) > 220.0f) continue;

        glm::vec3 lp = lamp_pos[i];

        pl_pos[pl_count] = lp;
        pl_col[pl_count] = lamp_col;
        pl_rad[pl_count] = lamp_rad;

        float side = (lp.x > 0.0f) ? -1.0f : 1.0f;
        glm::vec3 dir = glm::normalize(glm::vec3(side * 0.9f, -1.0f, 0.05f));
        pl_dir[pl_count] = dir;

        pl_cos[pl_count] = std::cos(glm::radians(lamp_ang_deg));
        ++pl_count;
    }

    glUniform1i(loc_pl_count, pl_count);
    if (pl_count > 0) {
        glUniform3fv(loc_pl_pos, pl_count, glm::value_ptr(pl_pos[0]));
        glUniform3fv(loc_pl_col, pl_count, glm::value_ptr(pl_col[0]));
        glUniform1fv(loc_pl_rad, pl_count, pl_rad);
        glUniform3fv(loc_pl_dir, pl_count, glm::value_ptr(pl_dir[0]));
        glUniform1fv(loc_pl_cos, pl_count, pl_cos);
    }

    // дорога

    glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);
    glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
    glm::mat4 m_tr = glm::mat4(1.0f);
    glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_tr));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, track_model.tex);
    draw_mesh(track_model.m);

    // обочина
    glBindTexture(GL_TEXTURE_2D, grass_model.tex);
    glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
    glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);

    // правая сторона
    {
        glm::mat4 m_gr = glm::mat4(1.0f);
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_gr));
        draw_mesh(grass_model.m);
    }

    // левая сторона
    {
        glm::mat4 m_gr = glm::mat4(1.0f);
        m_gr = glm::scale(m_gr, glm::vec3(-1.0f, 1.0f, 1.0f));
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_gr));
        draw_mesh(grass_model.m);
    }

    // машина
    glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);
    glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
    glm::mat4 m_car = glm::mat4(1.0f);
    m_car = glm::translate(m_car, car.pos + glm::vec3(0.0f, car_y_off, 0.0f));
    m_car = glm::rotate(m_car, car.yaw, glm::vec3(0, 1, 0));
    m_car = glm::rotate(m_car, glm::radians(0.0f), glm::vec3(0, 1, 0));
    m_car = glm::scale(m_car, glm::vec3(car_scale));
    if (!car_model.parts.empty()) {
        for (const auto &part : car_model.parts) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, part.tex ? part.tex : track_model.tex);

            glUniform3fv(loc_mat_col, 1, glm::value_ptr(part.kd));
            glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);

            glm::mat4 base_m = m_car;

            if (part.name.find("Wheel") != std::string::npos || part.name.find("Rims") != std::string::npos ||
                part.name.find("rim") != std::string::npos || part.name.find("wheel") != std::string::npos) {

                glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(base_m));
                draw_mesh(part.m);

            } else {
                glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(base_m));
                draw_mesh(part.m);
            }
        }
    } else {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, car_model.tex);
        glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
        glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_car));
        draw_mesh(car_model.m);
    }

    // стопы
    {
        glm::vec3 tail_base = car.pos + up * tail_y - fwd * tail_back;
        glm::vec3 tail_emis = tail_col_idle * 0.7f;
        if (car.brake) tail_emis = tail_col_brake;

        glBindTexture(GL_TEXTURE_2D, track_model.tex);

        draw_light_sphere(v * m_car, tail_base + right * tail_side, 0.06f, tail_emis,
                          loc_m, loc_emis, loc_mat_col);
        draw_light_sphere(v * m_car, tail_base - right * tail_side, 0.06f, tail_emis,
                          loc_m, loc_emis, loc_mat_col);
    }

    // трафик
    glBindTexture(GL_TEXTURE_2D, traffic_car_model.tex);
    for (const auto& tc : traffic) {
        glm::mat4 m = glm::mat4(1.0f);
        m = glm::translate(m, tc.pos);

        float yaw_tc = (tc.dir > 0.0f) ? 0.0f : glm::radians(180.0f);
        m = glm::rotate(m, yaw_tc, glm::vec3(0, 1, 0));
        m = glm::scale(m, glm::vec3(1.4f, 1.4f, 1.8f));

        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m));
        glUniform3fv(loc_mat_col, 1, glm::value_ptr(tc.color));
        glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);

        draw_mesh(traffic_car_model.m);
    }

    // фонари столбы
    glBindTexture(GL_TEXTURE_2D, track_model.tex);
    for (const auto& lp : lamp_pos) {
        glm::mat4 m_lamp = glm::mat4(1.0f);
        m_lamp = glm::translate(m_lamp, glm::vec3(lp.x, 0.0f, lp.z));

        glUniform3f(loc_mat_col, 0.12f, 0.12f, 0.12f);
        glUniform3f(loc_emis, 0.40f, 0.36f, 0.12f);

        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_lamp));
        draw_mesh(lamp_mesh);
    }

    // отбойник
    glBindTexture(GL_TEXTURE_2D, rail_model.tex);
    glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
    glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);

    float min_z = -track_half_len + rail_seg_len * 0.5f;
    float max_z =  track_half_len - rail_seg_len * 0.5f;
    float rail_x = track_half_w * 1.02f;

    for (float z = min_z; z <= max_z; z += rail_seg_len) {
        glm::mat4 m_l = glm::mat4(1.0f);
        m_l = glm::translate(m_l, glm::vec3( rail_x, 0.0f, z));
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_l));
        draw_mesh(rail_model.m);

        glm::mat4 m_r = glm::mat4(1.0f);
        m_r = glm::translate(m_r, glm::vec3(-rail_x, 0.0f, z));
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_r));
        draw_mesh(rail_model.m);
    }

    // деревюя
    for (const auto& tp : tree_pos) {
        glm::mat4 m = glm::mat4(1.0f);
        m = glm::translate(m, tp);
        m = glm::scale(m, glm::vec3(tree_scale));

        // ствол
        glBindTexture(GL_TEXTURE_2D, tree_trunk_model.tex);
        glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
        glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);
        glm::mat4 m_tr2 = m;
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_tr2));
        draw_mesh(tree_trunk_model.m);

        // листики
        glBindTexture(GL_TEXTURE_2D, tree_leaves_model.tex);
        glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
        glUniform3f(loc_emis, 0.02f, 0.04f, 0.02f); // лёгкое свечение листвы
        glm::mat4 m_lv = m;
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_lv));
        draw_mesh(tree_leaves_model.m);
    }

    // скамейки
    for (const auto& bp : bench_pos) {
        glm::mat4 m = glm::mat4(1.0f);
        m = glm::translate(m, bp);

        // поворот к трассе
        float angle = (bp.x > 0.0f) ? glm::radians(90.0f) : glm::radians(-90.0f);
        m = glm::rotate(m, angle, glm::vec3(0, 1, 0));
        m = glm::scale(m, glm::vec3(bench_scale));

        // дерево
        glBindTexture(GL_TEXTURE_2D, bench_wood_model.tex);
        glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
        glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);
        glm::mat4 m_wood = m;
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_wood));
        draw_mesh(bench_wood_model.m);

        // метал
        glBindTexture(GL_TEXTURE_2D, bench_metal_model.tex);
        glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
        glUniform3f(loc_emis, 0.0f, 0.0f, 0.0f);
        glm::mat4 m_met = m;
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m_met));
        draw_mesh(bench_metal_model.m);
    }

    // кусты
    glBindTexture(GL_TEXTURE_2D, bush_model.tex);
    for (const auto& bp : bush_pos) {
        glm::mat4 m = glm::mat4(1.0f);
        m = glm::translate(m, bp);
        m = glm::scale(m, glm::vec3(bush_scale));

        glUniform3f(loc_mat_col, 1.0f, 1.0f, 1.0f);
        glUniform3f(loc_emis, 0.01f, 0.03f, 0.01f);
        glUniformMatrix4fv(loc_m, 1, GL_FALSE, glm::value_ptr(m));
        draw_mesh(bush_model.m);
    }
}

// обнновление сцены
void run_loop() {
    last_time = (float)glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        float now = (float)glfwGetTime();
        dt = now - last_time;
        last_time = now;

        glfwPollEvents();

        if (key_r) {
            reset_car();
            key_r = false;
        }

        if (key_esc) {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }

        upd_car(dt);
        upd_traffic(dt);
        upd_cam(dt);
        draw_scene();
        upd_gate(dt);
        draw_gate();

        glfwSwapBuffers(win);
    }
}

// main
int main() {
    if (!init_gl() or !load_content()) return -1;

    run_loop();
    glfwTerminate();

    return 0;
}
