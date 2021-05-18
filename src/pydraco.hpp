#ifndef DVIDUTILS_PYDRACO_HPP
#define DVIDUTILS_PYDRACO_HPP

#include <cstdint>
#include <tuple>
#include <algorithm>
#include <iostream>

#include "draco/mesh/mesh.h"
#include "draco/compression/encode.h"
#include "draco/compression/decode.h"

#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor-python/pytensor.hpp"

using std::uint32_t;
using std::size_t;

namespace py = pybind11;

typedef xt::pytensor<float, 2> vertices_array_t;
typedef xt::pytensor<float, 2> normals_array_t;
typedef xt::pytensor<uint32_t, 2> faces_array_t;
typedef xt::pytensor<int, 1> coords_t;
/*
-DCMAKE_BUILD_TYPE=Debug     -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -DXTENSOR_ENABLE_ASSERT=ON"     -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" -DCMAKE_CXX_FLAGS=-I/groups/scicompsoft/home/ackermand/miniconda3/envs/multiresolution/include/python3.7m/pybind11
*/
struct Quantizer {
  // Constructs a quantizer.
  //
  // \param fragment_origin Minimum input vertex position to represent.
  // \param fragment_shape The inclusive maximum vertex position to represent
  //     is `fragment_origin + fragment_shape`.
  // \param input_origin The offset to add to input vertices before quantizing
  //     them within the `[fragment_origin, fragment_origin+fragment_shape]`
  //     range.
  // \param num_quantization_bits The number of bits to use for quantization.
  //     A value of `0` for coordinate `i` corresponds to `fragment_origin[i]`,
  //     while a value of `2**num_quantization_bits-1` corresponds to
  //     `fragment_origin[i]+fragment_shape[i]`.  Should be less than or equal
  //     to the number of bits in `VertexCoord`.
  Quantizer(coords_t const & fragment_shape, coords_t const & fragment_origin, int num_quantization_bits) {
      //assumes has been scaled between 0 and 1
        for (int i = 0; i < 3; ++i) {
            upper_bound[i] =
                static_cast<double>(std::numeric_limits<uint32_t>::max() >>
                                    (sizeof(uint32_t) * 8 - num_quantization_bits));
            fragment_shape_double[i] = static_cast<double>(fragment_shape[i]);
            //scale[i] = upper_bound[i] / static_cast<float>(fragment_shape[i]);
            offset[i] = fragment_origin[i] ;//mesh_origin[i] - fragment_origin[i] + 0.5 / scale[i];
        }
  }

  // Maps an input vertex position `v_pos`.
  std::array<uint32_t, 3> operator()(std::array<float, 3>& v_pos) {
    for (int i = 0; i < 3; ++i) {
      output[i] = static_cast<uint32_t>(std::min(
          //might need to further fix this due to rounding artifacts which set eg a value to 511 when it should be 512. that is why i moved scale out here
          upper_bound[i], std::max(0.0, (v_pos[i] - offset[i]) * upper_bound[i] / fragment_shape_double[i] + 0.5))); // Add 0.5 to round to nearest rather than round down.
    }
    return output;
  }

  std::array<uint32_t, 3> output;
  std::array<double, 3> offset;
  std::array<double, 3> scale;
  std::array<double, 3> upper_bound;
  std::array<double, 3> fragment_shape_double; 
};

// Defaults from the draco_encoder command-line tool.
int DEFAULT_COMPRESSION_LEVEL = 7;
int DEFAULT_POSITION_QUANTIZATION_BITS = 14;
int DEFAULT_NORMAL_QUANTIZATION_BITS = 10;
int DEFAULT_GENERIC_QUANTIZATION_BITS = 8;
bool DEFAULT_DO_CUSTOM = true;


// Encode the given vertices and faces arrays from python
// into a buffer (bytes object) encoded via draco.
//
// Special case: If faces is empty, an empty buffer is returned.
//
// Note: The vertices are expected to be passed in X,Y,Z order

py::bytes encode_faces_to_custom_drc_bytes( vertices_array_t const & vertices,
                                     normals_array_t const & normals,
                                     faces_array_t const & faces,
                                     coords_t const & fragment_shape,
                                     coords_t const & fragment_origin,
                                     int compression_level,
                                     int position_quantization_bits,
                                     int normal_quantization_bits,
                                     int generic_quantization_bits,
                                     bool do_custom)
{
    using namespace draco;
    DataType data_type = do_custom ? DT_UINT32 : DT_FLOAT32;

    auto vertex_count = vertices.shape()[0];
    auto normal_count = do_custom ? 0 :normals.shape()[0]; //FOR CUSTOM IGNORE NORMALS, SCREWS UP DECODING
    auto face_count = faces.shape()[0];

    Quantizer quantizer(fragment_shape, fragment_origin, position_quantization_bits);


    // Special case:
    // If faces is empty, an empty buffer is returned.
    if (face_count == 0)
    {
        return py::bytes();
    }
    
    int max_vertex = xt::amax(faces)();
    if (vertex_count < max_vertex+1)
    {
        throw std::runtime_error("Face indexes exceed vertices length");
    }
    
    if (normal_count > 0 and normal_count != vertex_count)
    {
        throw std::runtime_error("normals array size does not correspond to vertices array size");
    }

    draco::EncoderBuffer buf; // result

    // Release the GIL in the following scope.
    // (No python functions or data structures are touched in this scope)
    {
        py::gil_scoped_release nogil;

        Mesh mesh;
        mesh.set_num_points(vertex_count);
        mesh.SetNumFaces(face_count);
        
        // Init vertex attribute
        PointAttribute vert_att_template;
        vert_att_template.Init( GeometryAttribute::POSITION,    // attribute_type
                                nullptr,                        // buffer
                                3,                              // num_components
                                data_type,                     // data_type
                                false,                          // normalized
                                DataTypeLength(data_type) * 3, // byte_stride
                                0 );                            // byte_offset
        
        vert_att_template.SetIdentityMapping();

        // Add vertex attribute to mesh (makes a copy internally)
        int vert_att_id = mesh.AddAttribute(vert_att_template, true, vertex_count);
        mesh.SetAttributeElementType(vert_att_id, MESH_VERTEX_ATTRIBUTE);

        // Get a reference to the mesh's copy of the vertex attribute
        PointAttribute & vert_att = *(mesh.attribute(vert_att_id));

        // Load the vertices into the vertex attribute
        for (size_t vi = 0; vi < vertex_count; ++vi)
        {
            std::array<float, 3> v{{ vertices(vi, 0), vertices(vi, 1), vertices(vi, 2) }};
            if(do_custom){
                std::array<uint32_t,3> quantized_v = quantizer(v);
                vert_att.SetAttributeValue(AttributeValueIndex(vi), quantized_v.data());
            }
            else{
                vert_att.SetAttributeValue(AttributeValueIndex(vi), v.data());
            }
        }
        
        if (normal_count > 0)
        {
            // Init normal attribute
            PointAttribute norm_att_template;
            norm_att_template.Init( GeometryAttribute::NORMAL,      // attribute_type
                                    nullptr,                        // buffer
                                    3,                              // num_components
                                    DT_FLOAT32,                     // data_type
                                    false,                          // normalized
                                    DataTypeLength(DT_FLOAT32) * 3, // byte_stride
                                    0 );                            // byte_offset
            norm_att_template.SetIdentityMapping();

            // Add normal attribute to mesh (makes a copy internally)
            int norm_att_id = mesh.AddAttribute(norm_att_template, true, normal_count);
            mesh.SetAttributeElementType(norm_att_id, MESH_VERTEX_ATTRIBUTE);

            // Get a reference to the mesh's copy of the normal attribute
            PointAttribute & norm_att = *(mesh.attribute(norm_att_id));

            // Load the normals into the normal attribute
            for (size_t ni = 0; ni < normal_count; ++ni)
            {
                std::array<float, 3> n{{ normals(ni, 0), normals(ni, 1), normals(ni, 2) }};
                norm_att.SetAttributeValue(AttributeValueIndex(ni), n.data());
            }
        }
        
        // Load the faces
        for (size_t f = 0; f < face_count; ++f)
        {
            Mesh::Face face = {{ PointIndex(faces(f, 0)),
                                 PointIndex(faces(f, 1)),
                                 PointIndex(faces(f, 2)) }};

            for (auto vi : face)
            {
                assert(vi < vertex_count && "face has an out-of-bounds vertex");
            }

            mesh.SetFace(draco::FaceIndex(f), face);
        }
        
        mesh.DeduplicateAttributeValues();
        mesh.DeduplicatePointIds();

        draco::Encoder encoder;

        int speed = 10 - compression_level;
        encoder.SetSpeedOptions(speed, speed);
        encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, position_quantization_bits);
        if(!do_custom) encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL,   normal_quantization_bits);
        encoder.SetAttributeQuantization(draco::GeometryAttribute::GENERIC,  generic_quantization_bits);

        encoder.EncodeMeshToBuffer(mesh, &buf);
    }
    
    // Safe to use python again now that the GIL is re-acquired.
    return py::bytes(buf.data(), buf.size());
}
py::bytes encode_faces_to_drc_bytes( vertices_array_t const & vertices,
                                     normals_array_t const & normals,
                                     faces_array_t const & faces,
                                     int compression_level,
                                     int position_quantization_bits,
                                     int normal_quantization_bits,
                                     int generic_quantization_bits)
{
    coords_t fragment_shape = xt::zeros<int>({3});
    coords_t fragment_origin = xt::zeros<int>({3});

    bool do_custom = false;
    return encode_faces_to_custom_drc_bytes(vertices, normals, faces, fragment_shape, fragment_origin, compression_level, position_quantization_bits, normal_quantization_bits, generic_quantization_bits, do_custom);
}

// Decode a draco-encoded buffer (given as a python bytes object)
// into a xtensor-python arrays for the vertices and faces
// (which are converted to numpy arrays on the python side).
//
// Special case: If drc_bytes is empty, return empty vertices and faces.
//
// Note: The vertexes are returned in X,Y,Z order.
std::tuple<vertices_array_t, normals_array_t, faces_array_t> decode_drc_bytes_to_faces( py::bytes const & drc_bytes )
{
    using namespace draco;
    
    // Special case:
    // If drc_bytes is empty, return empty vertices and faces.
    if (py::len(drc_bytes) == 0)
    {
        vertices_array_t::shape_type verts_shape = {{0, 3}};
        vertices_array_t vertices(verts_shape);

        normals_array_t::shape_type normals_shape = {{0, 3}};
        normals_array_t normals(normals_shape);
        
        faces_array_t::shape_type faces_shape = {{0, 3}};
        faces_array_t faces(faces_shape);

        return std::make_tuple( std::move(vertices), std::move(normals), std::move(faces) );
    }

    // Extract pointer to raw bytes (avoid copy)
    PyObject * pyObj = drc_bytes.ptr();
    char * raw_buf = nullptr;
    Py_ssize_t bytes_length = 0;
    PyBytes_AsStringAndSize(pyObj, &raw_buf, &bytes_length);

    DecoderBuffer buf;
    int point_count;
    typedef std::unique_ptr<Mesh> MeshPtr;
    MeshPtr pMesh;
    
    {
        // Release GIL while decoding the mesh in C++
        py::gil_scoped_release nogil;

        buf.Init( raw_buf, bytes_length );

        // Decode to Mesh
        Decoder decoder;

        auto geometry_type = decoder.GetEncodedGeometryType(&buf).value();
        if (geometry_type != TRIANGULAR_MESH)
        {
            throw std::runtime_error("Buffer does not appear to be a mesh file. (Is it a pointcloud?)");
        }

        // Wrap bytes in a DecoderBuffer
        StatusOr<MeshPtr> decoded = decoder.DecodeMeshFromBuffer(&buf);
        if (!decoded.status().ok())
        {
            std::ostringstream ss;
            ss << "draco::Decoder::DecodeMeshFromBuffer() returned bad status: " << decoded.status();
            throw std::runtime_error(ss.str().c_str());
        }
        
        // This use of std::move feels like an ugly hack to workaround the fact
        // that StatusOr does not declare the following member:
        //   T & value() & { return value_; }
        // ... it declares a const version of it, which doesn't help us...
        pMesh = std::move(decoded).value();
        
        // Strangely, encoding a mesh may cause it to have duplicate point ids,
        // so we should de-duplicate them after decoding.
        pMesh->DeduplicateAttributeValues();
        pMesh->DeduplicatePointIds();
    
        point_count = pMesh->num_points();
    }
    
    // Initialize Python arrays (with GIL re-aqcuired)
    
    // Vertices
    vertices_array_t::shape_type verts_shape = {{point_count, 3}};
    vertices_array_t vertices(verts_shape);
    
    // Normals
    const PointAttribute *const normal_att = pMesh->GetNamedAttribute(GeometryAttribute::NORMAL);
    normals_array_t::shape_type::value_type normal_count = 0;
    if (normal_att != nullptr)
    {
        // See Note below about why we don't use normal_att->size()
        normal_count = point_count;
    }
    
    normals_array_t::shape_type normals_shape = {{normal_count, 3}};
    normals_array_t normals(normals_shape);
    
    // Faces
    faces_array_t::shape_type faces_shape = {{pMesh->num_faces(), 3}};
    faces_array_t faces(faces_shape);

    {
        // Release GIL again while copying from pMesh into the arrays
        py::gil_scoped_release nogil;

        // Extract vertices
        const PointAttribute *const vertex_att = pMesh->GetNamedAttribute(GeometryAttribute::POSITION);
        if (vertex_att == nullptr)
        {
            throw std::runtime_error("Draco mesh appears to have no vertices.");
        }

        std::array<float, 3> vertex_value;
        for (PointIndex i(0); i < point_count; ++i)
        {
            if (!vertex_att->ConvertValue<float, 3>(vertex_att->mapped_index(i), &vertex_value[0]))
            {
                std::ostringstream ssErr;
                ssErr << "Error reading vertex " << i.value() << std::endl;
                throw std::runtime_error(ssErr.str());
            }
            vertices(i.value(), 0) = vertex_value[0];
            vertices(i.value(), 1) = vertex_value[1];
            vertices(i.value(), 2) = vertex_value[2];
        }

        // Extract normals (if any)
        if (normal_count > 0)
        {
            std::array<float, 3> normal_value;

            // Important:
            // We don't use normal_att->size(), because it might be smaller
            // than the number of vertices (if not all vertices had unique normals).
            // Instead, we loop over the POINT indices, mapping from point indices to normal entries.
            for (PointIndex i(0); i < normal_count; ++i)
            {
                if (!normal_att->ConvertValue<float, 3>(normal_att->mapped_index(i), &normal_value[0]))
                {
                    std::ostringstream ssErr;
                    ssErr << "Error reading normal for point " << i << std::endl;
                    throw std::runtime_error(ssErr.str());
                }
                normals(i.value(), 0) = normal_value[0];
                normals(i.value(), 1) = normal_value[1];
                normals(i.value(), 2) = normal_value[2];
            }
        }

        // Extract faces
        for (auto i = 0; i < pMesh->num_faces(); ++i)
        {
            auto const & face = pMesh->face(FaceIndex(i));
            faces(i, 0) = face[0].value();
            faces(i, 1) = face[1].value();
            faces(i, 2) = face[2].value();
        }
    }

    return std::make_tuple( std::move(vertices), std::move(normals), std::move(faces) );
}

#endif
