Shader "PBR_IBL"{
    Properties{
        _Color("Color", Color) = (1,1,1,1)                       // 颜色
        _MainTex("Albedo (RGB)", 2D) = "white" {}                // 反照率
        _MetallicGlossMap("Metallic", 2D) = "white"{}            // 金属贴图（r金属度，a光滑度）
        _BumpMap("Normal Map", 2D) = "bump" {}                   // 法线贴图
        _OcclusionMap("Occlusion", 2D) = "white"{}               // 环境光遮蔽纹理
        _MetallicStrength("Metallic Strength", Range(0,1)) = 1   // 金属强度
        _GlossStrength("Gloss Strength", Range(0,1)) = 0.5       // 光滑强度
        _BumpScale("Bump Scale", Float) = 1                      // 法线影响
        _EmissionColor("Emission Color", Color) = (0,0,0,0)      // 自发光颜色
        _EmissionMap("Emission Map", 2D) = "white"{}             // 自发光贴图
    }

    CGINCLUDE

        #include "UnityCG.cginc"
        #include "Lighting.cginc"
        #include "AutoLight.cginc"

    //间接光计算
    
        // 顶点光照辅助函数，计算顶点级别的环境光或光照贴图uv信息
        // 参数：主uv（用于静态光照贴图）；动态光照贴图uv；顶点世界位置；顶点世界法线
        inline half4 VertexGI(float2 uv1, float2 uv2, float3 worldPos, float3 worldNormal){
            
            half4 ambientOrLightmapUV = 0;
            // 如果开启光照贴图，计算光照贴图的uv
            #ifdef LIGHTMAP_ON
                ambientOrLightmapUV.xy = uv1.xy * unity_LightmapST.xy + unity_LightmapST.zw;
            #elif UNITY_SHOULD_SAMPLE_SH
                // 计算非重要的顶点光照
                #ifdef VERTEXLIGHT_ON
                    // 计算4个顶点光照（UnityCG.cginc）
                    // - 自动提供4个最重要的点光源信息
                    ambientOrLightmapUV.rgb = Shade4PointLights(
                        unity_4LightPosX0, unity_4LightPosY0, unity_4LightPosZ0,
                        unity_LightColor[0].rgb, unity_LightColor[1].rgb, unity_LightColor[2].rgb, unity_LightColor[3].rgb,
                        unity_4LightAtten0, worldPos, worldNormal);
                #endif
                // 计算球谐光照（UnityCG.cginc）
                ambientOrLightmapUV.rgb += ShadeSH9(half4(worldNormal,1));
            #endif

            // 实时全局光照：动态生成光照贴图uv
            #ifdef DYNAMICLIGHTMAP_ON
                ambientOrLightmapUV.zw = uv2.xy * unity_DynamicLightmapST.xy + unity_DynamicLightmapST.zw;
            #endif

            return ambientOrLightmapUV;
        }

        // 计算间接漫反射光照
        // 参数：环境光或光照贴图uv；环境光遮蔽
        inline half3 IndirectDiffuse(half4 ambientOrLightmapUV, half occlusion){
            half3 indirectDiffuse = 0;

            // SH球谐光照 模拟漫反射
            // - 用于动态物体或不适用光照贴图的物体
            #if UNITY_SHOULD_SAMPLE_SH
                indirectDiffuse = ambientOrLightmapUV.rgb;
            #endif
            // 静态光照贴图（预烘焙）
            #ifdef LIGHTMAP_ON
                indirectDiffuse = DecodeLightmap(UNITY_SAMPLE_TEX2D(unity_Lightmap,ambientOrLightmapUV.xy));
            #endif
            // 动态光照贴图（实时全局光照）
            #ifdef DYNAMICLIGHTMAP_ON
                indirectDiffuse += DecodeRealtimeLightmap(UNITY_SAMPLE_TEX2D(unity_DynamicLightmap, ambientOrLightmapUV.zw));
            #endif

            return indirectDiffuse * occlusion;
        }

        // 盒状投影反射
        // - 根据反射探针的盒子体积，对原始世界反射方向 worldRefDir 进行校正
        // - 默认cubemap反射贴图是无限远的，对于局部盒状区域会造成拉伸
        // 世界空间下的反射方向向量；当前像素/片元的世界位置；反射探针位置；反射盒子最小边界；反射盒子最大边界
        inline half3 BoxProjection(half3 worldRefDir, float3 worldPos, float4 cubemapCenter, float4 boxMin, float4 boxMax){
            
            //使if语句产生分支(HLSLSupport.cginc)
            UNITY_BRANCH
            //如果反射探头开启了BoxProjection选项，cubemapCenter.w > 0
            if(cubemapCenter.w > 0.0){
                half3 rbmax = (boxMax.xyz - worldPos) / worldRefDir;
                half3 rbmin = (boxMin.xyz - worldPos) / worldRefDir;
                half3 rbminmax = (worldRefDir.xyz > 0.0f) ? rbmax : rbmin;
                // 最早撞墙距离
                half fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
                
                // 转化坐标
                // - 现将当前点转移到cubemap中心为原点的空间
                // - 然后打到边界上
                worldPos -= cubemapCenter.xyz;
                worldRefDir = worldPos + worldRefDir * fa;
            }
            return worldRefDir;
        }

        // 基于粗糙度 从与滤波环境贴图（反射探头）中采样模糊高光
        // 参数：反射探头贴图；反射方向；粗糙度；反射探头hdr信息
        // - UNITY_ARGS_TEXCUBE (HLSLSupport.cginc),用来区别平台
        inline half3 SamplerReflectProbe(UNITY_ARGS_TEXCUBE(tex), half3 refDir, half roughness, half4 hdr){
            // 经验公式，对粗糙度进行 感知线性化重映射，使低粗糙度区域有更细腻的响应
            roughness = roughness * (1.7 - 0.7 * roughness);
            // 反射探针使用预滤波的立方体贴图，有多个mip level
            // mip 0：最清晰（镜面反射）
            // mip 5~6：最模糊（镜面反射）
            half mip = roughness * 6;

            // 指定mip级别的采样，结果是RGBM编码的颜色值
            half4 rgbm = UNITY_SAMPLE_TEXCUBE_LOD(tex, refDir, mip);

            // HDR解码（UnityCG.cginc）：从RGBM格式中解码为线性HDR颜色
            return DecodeHDR(rgbm, hdr);
        }

        // 间接高光反射
        // 参数：世界空间反射方向；当前像素/片元的世界位置；粗糙度；环境光遮蔽因子
        inline half3 IndirectSpecular(half3 refDir, float3 worldPos, half roughness, half occlusion){
            half3 specular = 0;
            // 对第一个反射探针进行Box Projection方向修正
            half3 refDir1 = BoxProjection(refDir, worldPos, unity_SpecCube0_ProbePosition, unity_SpecCube0_BoxMin, unity_SpecCube0_BoxMax);
            half3 ref1 = SamplerReflectProbe(UNITY_PASS_TEXCUBE(unity_SpecCube0), refDir1, roughness, unity_SpecCube0_HDR);

            // 反射探针的空间边界：需要混合两个探针
            UNITY_BRANCH
            if(unity_SpecCube1_BoxMin.w < 0.99999){
                half3 refDir2 = BoxProjection(refDir, worldPos, unity_SpecCube1_ProbePosition, unity_SpecCube1_BoxMin, unity_SpecCube1_BoxMax);
                half3 ref2 = SamplerReflectProbe(UNITY_PASS_TEXCUBE_SAMPLER(unity_SpecCube1, unity_SpecCube0), refDir2, roughness, unity_SpecCube1_HDR);
                specular = lerp(ref2, ref1, unity_SpecCube0_BoxMin.w);
            }
            else{
                specular = ref1;
            }
            return specular * occlusion;
        }
    
    // BRDF
        // G:几何遮蔽函数
        // 参数：法线n与光线方向l余弦；法线n与视角方向v余弦；材质粗糙度
        inline half SchlickGGX_Unity5(half nl, half nv, half roughness){
            half k = roughness * roughness / 2;
            half lambdaV = (nv * (1 - k) + k);
            half lambdaL = (nl * (1 - k) + k);
            return 1.0f / (lambdaV * lambdaL);
        }
        inline half SchlickGGX_UE4(half nl, half nv, half roughness){
            half k = (roughness + 1) * (roughness + 1) / 8;
            // half k = (roughness* roughness) / 2;
            half G_v = nv / (nv * (1 - k) + k);
            half G_l = nl / (nl * (1 - k) + k);

            return G_v * G_l;
        }

        // D：法线分布函数
        // 参数：法线n与半程向量h的余弦；粗糙度
        inline half GGXTerm(half nh, half roughness){
            half a = roughness * roughness;
            half a2 = a * a;
            half d = (nh * nh) * (a2 - 1.0f) + 1.0f;
            return a2 * UNITY_INV_PI / (d * d + 1e-5f);
        }

        // F：菲涅尔
		// 参数：基础反射率；法线v与半程向量h的余弦
        inline half3 SchlickFresnel(half3 F0, half hv){
            return F0 + (1 - F0) * pow(1 - hv, 5);
        }

        // burley 漫反射模型
        // 参数：法线n与视角方向v的余弦；法线n与光线方向l的余弦；光线方向l与半程向量h的余弦；粗糙度；基础反射率
        inline half3 BurleyDiffuseTerm(half nv, half nl, half lh, half roughness, half3 baseColor){
            half Fd90 = 0.5f + 2 * roughness * lh * lh;
            return baseColor * UNITY_INV_PI * (1 + (Fd90 - 1) * pow(1 - nl, 5)) * (1 + (Fd90 - 1) * pow(1 - nv, 5));
        }

        // lambert 漫反射模型
        inline half3 LambertDiffuseTerm(half3 baseColor){
            return baseColor * UNITY_INV_PI;
        }

        // 用于间接镜面反射中的 Fresnel 控制
        // 参数：基础反射率c0,；斜角极限反射率c1；法线n与观察方向v的余弦
        inline half3 ComputeFresnelLerp(half3 c0, half3 c1, half cosA){
            half t = pow(1 - cosA, 5);
            return lerp(c0, c1, t);
        }

    ENDCG
    SubShader{
        Tags{"RenderType" = "Opaque"}
        pass{
            Tags{"LightMode" = "ForwardBase"}
            CGPROGRAM
            #pragma target 3.0

            #pragma multi_compile_fwdbase
            #pragma multi_compile_fog

            #pragma vertex vert
            #pragma fragment frag

            half4 _Color;
			sampler2D _MainTex;
			float4 _MainTex_ST;
			sampler2D _MetallicGlossMap;
			sampler2D _BumpMap;
			sampler2D _OcclusionMap;
			half _MetallicStrength;
			half _GlossStrength;
			float _BumpScale;
			half4 _EmissionColor;
			sampler2D _EmissionMap;

            struct a2v
			{
				float4 vertex : POSITION;
				float3 normal : NORMAL;
				float4 tangent :TANGENT;
				float2 texcoord : TEXCOORD0;
				float2 texcoord1 : TEXCOORD1;
				float2 texcoord2 : TEXCOORD2;
			};
            
            struct v2f
			{
				float4 pos : SV_POSITION;
				float2 uv : TEXCOORD0;
				float4 TtoW0 : TEXCOORD1;
				float4 TtoW1 : TEXCOORD2;
				float4 TtoW2 : TEXCOORD3;//xyz 存储着 从切线空间到世界空间的矩阵，w存储着世界坐标
                half4 ambientOrLightmapUV : TEXCOORD4;//存储环境光或光照贴图的UV坐标
				SHADOW_COORDS(5) //定义阴影所需要的变量(AutoLight.cginc)
				UNITY_FOG_COORDS(6) //定义雾效所需要的变量(UnityCG.cginc)
			};

            v2f vert(a2v v){
                v2f o;
                UNITY_INITIALIZE_OUTPUT(v2f, o); //初始化结构体数据，定义在HLSLSupport.cginc

                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);

                float3 worldPos = mul(unity_ObjectToWorld, v.vertex);
                half3 worldNormal = UnityObjectToWorldNormal(v.normal);
                half3 worldTangent = UnityObjectToWorldDir(v.tangent);
                half3 worldBinormal = cross(worldNormal, worldTangent) * v.tangent.w;

                // 环境光照或环境贴图uv
                o.ambientOrLightmapUV = VertexGI(v.texcoord1, v.texcoord2, worldPos, worldNormal);

                o.TtoW0 = float4(worldTangent.x, worldBinormal.x, worldNormal.x, worldPos.x);
                o.TtoW1 = float4(worldTangent.y, worldBinormal.y, worldNormal.y, worldPos.y);
                o.TtoW2 = float4(worldTangent.z, worldBinormal.z, worldNormal.z, worldPos.z);

                //填充阴影所需要的参数(AutoLight.cginc)
				TRANSFER_SHADOW(o);
				//填充雾效所需要的参数(UnityCG.cginc)
				UNITY_TRANSFER_FOG(o,o.pos);

                return o;
            }

            half4 frag(v2f i) : SV_Target{
                // 数据准备
                float3 worldPos = float3(i.TtoW0.w, i.TtoW1.w, i.TtoW2.w);  
                half3 albedo = tex2D(_MainTex, i.uv).rgb * _Color.rgb;      // 反照率（漫反射）
                half2 metallicGloss = tex2D(_MetallicGlossMap, i.uv).ra;         
                half metallic = metallicGloss.x * _MetallicStrength;         // 金属度
                half roughness = 1 - metallicGloss.y * _GlossStrength;       // 粗糙度
                half occlusion = tex2D(_OcclusionMap, i.uv).g;               // 环境光遮挡

                // 法线（世界空间）
                half3 normalTangent = UnpackNormal(tex2D(_BumpMap, i.uv));
                normalTangent.xy *= _BumpScale;
                normalTangent.z = sqrt(1.0 - saturate(dot(normalTangent.xy, normalTangent.xy)));
                half3 worldNormal = normalize(half3(dot(i.TtoW0.xyz, normalTangent),
                                                    dot(i.TtoW1.xyz, normalTangent),
                                                    dot(i.TtoW2.xyz, normalTangent)));

                half3 lightDir = normalize(UnityWorldSpaceLightDir(worldPos));  // 光线方向WS(UnityCG.cginc)
                half3 viewDir = normalize(UnityWorldSpaceViewDir(worldPos));    // 视线方向WS(UnityCG.cginc)
                half3 refDir = reflect(-viewDir, worldNormal);                  // 反射方向WS

                // 自发光
                half3 emission = tex2D(_EmissionMap, i.uv).rgb * _EmissionColor;    

                UNITY_LIGHT_ATTENUATION(atten, i, worldPos);                //计算阴影和衰减(AutoLight.cginc)

                // 计算BRDF需要的项
                half3 halfDir = normalize(lightDir + viewDir);
                half nv = saturate(dot(worldNormal, viewDir));
                half nl = saturate(dot(worldNormal, lightDir));
                half nh = saturate(dot(worldNormal, halfDir));
                half lh = saturate(dot(lightDir, halfDir));
                half lv = saturate(dot(lightDir, viewDir));

                // 镜面反射率
                // - 纯金属只有镜面反射，所以金属度代表了镜面反射程度
                // - unity_ColorSpaceDielectricSpec.a 非金属材质的反射率（默认是 0.04）
                half3 specColor = lerp(unity_ColorSpaceDielectricSpec.rgb, albedo, metallic);       // 镜面反射颜色
                half oneMinusReflectivity = (1 - metallic) * unity_ColorSpaceDielectricSpec.a;
                // 漫反射率
                half3 diffColor = albedo * oneMinusReflectivity;                                   // 漫反射颜色    
                
            // 【间接光】
                // - 间接漫反射：SH球谐光照 or 光照贴图 or 实时光照贴图
                // - 间接镜面反射：采样反射探针cubemao（低mipLevel）
                half3 indirectDiffuse = IndirectDiffuse(i.ambientOrLightmapUV, occlusion);   
                half3 indirectSpecular = IndirectSpecular(refDir, worldPos, roughness, occlusion);
                // 计算掠射角时反射率（近似菲涅尔）
                half grazingTerm= saturate((1 - roughness) + (1 - oneMinusReflectivity));
                // 间接光镜面反射
                indirectSpecular *= ComputeFresnelLerp(specColor, grazingTerm, nv);
                // 间接光漫反射
                indirectDiffuse *= diffColor;

            // 【BRDF】直射光

                // BRDF 高光反射项
                half G = SchlickGGX_UE4 (nl, nv, roughness);              // G几何遮蔽函数：（Unity）
                // half G = SchlickGGX_Unity5(nl,nv,roughness);           
                half D = GGXTerm(nh, roughness);                          // D法线分布函数
                half3 F = SchlickFresnel(specColor, lh);                     // F菲涅尔
                half3 specularTerm = G * D * F / (4.0 * max(nv * nl, 0.001)); 

                // BRDF 漫反射项
                half3 diffuseTerm = BurleyDiffuseTerm(nv, nl, lh, roughness, diffColor);
                // half3 diffuseTerm = LambertDiffuseTerm(diffColor);

                // 【反射方程】
                half3 L_o = UNITY_PI * (diffuseTerm + specularTerm) * _LightColor0.rgb * nl;

                // 环境光
                float3 ambient = UNITY_LIGHTMODEL_AMBIENT.rgb * albedo;

                // 最终颜色
                // half3 color = L_o * atten + ambient+ emission;
                half3 color = L_o * atten + indirectDiffuse + indirectSpecular + emission;
               
                //雾效(UnityCG.cginc)
				UNITY_APPLY_FOG(i.fogCoord, color.rgb);

                return half4(color,1);
            }
            ENDCG
        }
    }
    FallBack "VertexLit"
}