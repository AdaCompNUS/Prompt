

class Pushing : public Scene {

public:
    Pushing(const char *name) :
            Scene(name) {}


public:

    char *make_path(char *full_path, std::string path) {
        strcpy(full_path, getenv("PYFLEXROOT"));
        strcat(full_path, path.c_str());
        return full_path;
    }


    virtual void Initialize(py::array_t<float> scene_params, int thread_idx = 0) {

        srand(time(NULL) + thread_idx);
        g_sceneLower = Vec3(-0.5f, 0.0f, 0.0f);
        g_sceneUpper = Vec3(1.0f, 2.0f, 0.6f);


        float radius = float(thread_idx) / 100000.0f;

        g_params.radius = radius;
        g_params.dynamicFriction = 0.5f;
        //g_params.staticFriction = 1.0f;
//        g_params.dynamicFriction = 0.75f;
        //g_params.particleFriction = 1;
        g_params.dissipation = 0.0f;
        g_params.numIterations = 8;
        g_params.viscosity = 0.0f;
        g_params.drag = 0.0f;
        g_params.lift = 0.0f;
        g_params.collisionDistance = radius * 0.5f;

        g_windStrength = 0.0f;
        g_numSubsteps = 2;
//        g_numSubsteps = 1;
        g_lightDistance *= 1.5f;


        createRigid(scene_params);
    }

    void createRigid(py::array_t<float> positions) {

        if (g_buffers->rigidIndices.empty()) {
            g_buffers->rigidOffsets.push_back(0);
        }

        auto buf = positions.request();
        auto ptr = (float *) buf.ptr;


        for (size_t i = 0; i < (int) (positions.size() / 4); i++) {

            g_buffers->rigidIndices.push_back(int(g_buffers->positions.size()));
            //g_buffers->rigidLocalNormals.push_back(Vec4(n, d));
            g_buffers->positions.push_back(Vec4(ptr[i * 4], ptr[i * 4 + 1], ptr[i * 4 + 2], ptr[i * 4 + 3]));
            g_buffers->velocities.push_back(Vec3(0, 0, 0));
            g_buffers->phases.push_back(NvFlexMakePhase(1, 0)); // rigid objs

        }
        g_buffers->rigidCoefficients.push_back(1.0f); //stiffness
//        g_buffers->rigidCoefficients.push_back(0.9f); //stiffness
        g_buffers->rigidOffsets.push_back(int(g_buffers->rigidIndices.size()));
    }


};
